"""SuperSLIM inference (NumPy only).

SuperSLIM is a compact descendant of SLIM (Rasmussen Lab, MIT licence); it is an independent project and is not affiliated with the SLIM authors.

A SuperSLIM model file (``.npz``) holds, for each *learned system* (a cell line or condition), a small head that predicts the mean transcriptome response to knocking
down a gene: ``prediction = base + scale * (latent @ basis)``. The latent vector is predicted from the gene's catalog features (4-bit table) and from the responses of the
same gene in other systems; the head is a ridge-linear map mixed with a small INT8 MLP. Output unit: change of the mean expression versus control cells (log1p counts per
10,000 for most systems). A prediction is only meaningful for the systems in the model file; there is no transfer to unseen cell lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

CTRL_SCALE = 1.0 / 4096.0


def _names(blob: np.ndarray) -> np.ndarray:
    return np.array(bytes(blob).decode().split("\n"))


def _unpack_q4(packed: np.ndarray, scale: np.ndarray, shape) -> np.ndarray:
    hi = (packed >> 4).astype(np.int16) - 8
    lo = (packed & 15).astype(np.int16) - 8
    q = np.empty(packed.size * 2, np.int16)
    q[0::2] = hi
    q[1::2] = lo
    return q.reshape(shape).astype(np.float32) * scale.astype(np.float32)


@dataclass
class Prediction:
    """Predicted responses. ``values[i, j]`` is the change of gene ``output_genes[j]`` when ``genes[i]`` is perturbed."""

    line: str
    genes: List[str]
    output_genes: np.ndarray
    values: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {g: self.values[i] for i, g in enumerate(self.genes)}

    def top(self, gene: str, n: int = 20, direction: str = "down") -> List[tuple]:
        """The ``n`` output genes with the strongest predicted decrease (``direction='down'``) or increase (``'up'``) for ``gene``."""
        v = self.values[self.genes.index(gene)]
        order = np.argsort(v) if direction == "down" else np.argsort(-v)
        return [(str(self.output_genes[j]), float(v[j])) for j in order[:n]]


class SuperSLIM:
    def __init__(self, arrays: Dict[str, np.ndarray]):
        z = self.z = dict(arrays)
        if "format" not in z or not str(z["format"]).startswith("superslim-open"):
            raise ValueError("Not a SuperSLIM model file (expected format 'superslim-open-v*').")
        self.catalog = _names(z["catalog_b"])
        self._lookup = {g: i for i, g in enumerate(self.catalog)}
        self._X = _unpack_q4(z["x_q4"], z["x_scale4"], (len(self.catalog), int(z["x_dim"])))
        self._systems = [str(x) for x in _names(z["systems_b"])]
        self._sources = [str(x) for x in _names(z["sources_b"])]
        self._out: Dict[str, tuple] = {}
        for n in self._systems:
            lg = _names(z[f"sys/{n}/genes_b"])
            self._out[n] = (lg, {g: i for i, g in enumerate(lg)})
        self._cx = {}
        for n in self._sources:
            idx = z[f"cx/{n}/idx"].astype(int)
            self._cx[n] = ({int(g): i for i, g in enumerate(idx)}, z[f"cx/{n}/codes"].astype(np.float32) * float(z[f"cx/{n}/scale"]) / z[f"cx/{n}/std"])

    # ------------------------------------------------------------------ loading / introspection
    @classmethod
    def load(cls, path: "str | Path") -> "SuperSLIM":
        with np.load(str(path), allow_pickle=False) as f:
            return cls({k: f[k] for k in f.files})

    @property
    def lines(self) -> List[str]:
        """Learned systems available in this file."""
        return list(self._systems)

    def genes(self) -> List[str]:
        """Input genes (catalog) that can be perturbed."""
        return [str(g) for g in self.catalog]

    def supports(self, gene: str) -> bool:
        return gene in self._lookup

    def output_genes(self, line: str) -> np.ndarray:
        self._check_line(line)
        return self._out[line][0]

    def info(self) -> Dict[str, object]:
        """Summary of the loaded model: systems, number of input genes, number of output genes per system."""
        return {"systems": self.lines, "n_input_genes": int(len(self.catalog)), "n_output_genes": {L: int(len(self.output_genes(L))) for L in self.lines}}

    def __repr__(self) -> str:
        return f"SuperSLIM(systems={self.lines}, input_genes={len(self.catalog)})"

    def _check_line(self, line: str) -> None:
        if line not in self._out:
            raise ValueError(f"Unknown system '{line}'. Available: {', '.join(self.lines)}")

    # ------------------------------------------------------------------ prediction
    def _x(self, names: Sequence[str]) -> np.ndarray:
        miss = [g for g in names if g not in self._lookup]
        if miss:
            raise ValueError("Unsupported input gene (no feature row): " + ", ".join(miss[:5]))
        return self._X[[self._lookup[g] for g in names]]

    def _mlp(self, prefix: str, Xs: np.ndarray) -> np.ndarray:
        z = self.z
        p = prefix + "_"
        outs = []
        for s in range(int(z[p + "mlp_n"])):
            x = Xs.astype(np.float32)
            for layer in range(3):
                W = z[f"{p}mlp{s}_w{layer}"].astype(np.float32) * z[f"{p}mlp{s}_s{layer}"][:, None]
                x = x @ W.T + z[f"{p}mlp{s}_b{layer}"].astype(np.float32)
                if layer < 2:
                    x = np.maximum(x, 0)
            outs.append(x)
        return np.mean(outs, axis=0)

    def predict(self, genes: Iterable[str], line: str) -> Prediction:
        """Predict the response of ``line`` to perturbing each gene in ``genes``."""
        names = list(genes)
        self._check_line(line)
        z = self.z
        p = f"sys/{line}/"
        F = [self._x(names)]
        for s in [str(x) for x in _names(z[p + "sources_b"])]:
            lookup, codes = self._cx[s]
            C = np.zeros((len(names), codes.shape[1]), np.float32)
            av = np.zeros((len(names), 1), np.float32)
            for i, g in enumerate(names):
                j = lookup.get(self._lookup[g])
                if j is not None:
                    C[i] = codes[j]
                    av[i, 0] = 1.0
            F += [C, av]
        F = np.column_stack(F).astype(np.float32)
        Xs = (F - z[p + "x_mean"]) / z[p + "x_sd"] * z[p + "x_scale"]
        a = float(z[p + "alpha"])
        Z = (1 - a) * (Xs @ z[p + "W"]) + a * self._mlp(f"sys/{line}", Xs)
        B = z[p + "basis_q"].astype(np.float32) * z[p + "basis_scale"][:, None]
        raw = (z[p + "base16"].astype(np.float32) + float(z[p + "s"]) * (Z @ B)).astype(np.float32)
        if bool(z[p + "has_control"]):  # expression (control + change) cannot go below zero
            ctrl = z[p + "control16"].astype(np.float32) * np.float32(CTRL_SCALE)
            raw = np.maximum(raw + ctrl, 0) - ctrl
        lg, gpos = self._out[line]
        own = float(z[p + "own_value"])
        for i, g in enumerate(names):  # the perturbed gene itself is set to a constant
            j = gpos.get(g)
            if j is not None:
                raw[i, j] = own
        return Prediction(line, names, lg, raw)

    def compare(self, gene: str, lines: Optional[Sequence[str]] = None) -> Dict[str, Prediction]:
        """The same gene's predicted response in several systems (each in its own output genes and units)."""
        return {L: self.predict([gene], L) for L in (lines or self.lines)}
