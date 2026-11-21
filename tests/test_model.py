"""Tests on a tiny synthetic model file (no real model needed)."""
import numpy as np
import pytest

from superslim import SuperSLIM
from superslim.cli import main as cli_main

N_CAT, DX, K, H, NO, NCODE = 20, 6, 3, 5, 9, 4
GOLD_SUM = 19.530868530273438


def _blob(names):
    return np.frombuffer("\n".join(names).encode(), np.uint8).copy()


def _pack_q4(X):
    sc = np.maximum(np.abs(X).max(0) / 7, 1e-8).astype(np.float32)
    q = (np.clip(np.rint(X / sc), -7, 7).astype(np.int16) + 8).astype(np.uint8).ravel()
    return ((q[0::2] << 4) | q[1::2]).astype(np.uint8), sc


def make_open(seed=0):
    rng = np.random.default_rng(seed)
    cat = [f"G{i}" for i in range(N_CAT)]
    packed, sc = _pack_q4(rng.normal(size=(N_CAT, DX)).astype(np.float32))
    z = {"format": np.array("superslim-open-v1"), "catalog_b": _blob(cat), "x_q4": packed, "x_scale4": sc, "x_dim": np.int32(DX), "systems_b": _blob(["sysa"]), "sources_b": _blob(["sysa", "sysb"])}
    for n in ("sysa", "sysb"):
        z[f"cx/{n}/idx"] = rng.choice(N_CAT, 12, replace=False).astype(np.int32)
        z[f"cx/{n}/codes"] = rng.integers(-100, 100, (12, NCODE)).astype(np.int8)
        z[f"cx/{n}/scale"] = np.float32(0.01)
        z[f"cx/{n}/std"] = np.ones(NCODE, np.float32)
    d = DX + 1 * (NCODE + 1)  # one source: sysb
    p = "sys/sysa/"
    out = [f"O{i}" for i in range(NO - 2)] + ["G1", "G2"]
    z.update({p + "genes_b": _blob(out), p + "basis_q": rng.integers(-50, 50, (K, NO)).astype(np.int8), p + "basis_scale": np.full(K, 0.01, np.float32), p + "base16": (rng.normal(size=NO) * 0.01).astype(np.float16),
              p + "W": rng.normal(size=(d, K)).astype(np.float32) * 0.1, p + "x_mean": np.zeros(d, np.float32), p + "x_sd": np.ones(d, np.float32), p + "x_scale": np.ones(d, np.float32), p + "s": np.float32(1.5), p + "alpha": np.float32(0.25),
              p + "own_value": np.float32(-0.3), p + "has_control": np.array(True), p + "control16": np.rint(np.abs(rng.normal(size=NO)) * 0.2 * 4096).astype(np.int16), p + "sources_b": _blob(["sysb"]), "sys/sysa_mlp_n": np.int32(1)})
    for layer, (o, i) in enumerate([(H, d), (H, H), (K, H)]):
        z[f"sys/sysa_mlp0_w{layer}"] = rng.integers(-50, 50, (o, i)).astype(np.int8)
        z[f"sys/sysa_mlp0_s{layer}"] = np.full(o, 0.01, np.float32)
        z[f"sys/sysa_mlp0_b{layer}"] = np.zeros(o, np.float32)
    return z


def test_open_model_basics():
    m = SuperSLIM(make_open())
    assert m.lines == ["sysa"] and all(type(x) is str for x in m.lines) and "np.str_" not in repr(m) and len(m.genes()) == N_CAT
    r = m.predict(["G0", "G1", "G5"], "sysa")
    assert r.values.shape == (3, NO) and np.isfinite(r.values).all()
    assert np.array_equal(r.values, m.predict(["G0", "G1", "G5"], "sysa").values)


def test_open_model_floor_and_own_gene():
    m = SuperSLIM(make_open())
    r = m.predict([f"G{i}" for i in range(N_CAT)], "sysa")
    ctrl = m.z["sys/sysa/control16"].astype(np.float32) / 4096
    j = list(r.output_genes).index("G1")
    mask = np.ones(NO, bool)
    mask[[list(r.output_genes).index("G1"), list(r.output_genes).index("G2")]] = False
    assert (r.values[:, mask] + ctrl[mask] >= -1e-6).all()
    assert r.values[1, j] == pytest.approx(-0.3)


def test_open_model_unknown_system_and_gene():
    m = SuperSLIM(make_open())
    with pytest.raises(ValueError):
        m.predict(["G0"], "nope")
    with pytest.raises(ValueError):
        m.predict(["X_unknown"], "sysa")


def test_info_repr_and_golden_values():
    m = SuperSLIM(make_open())
    info = m.info()
    assert info["systems"] == ["sysa"] and info["n_input_genes"] == N_CAT and info["n_output_genes"]["sysa"] == NO
    r = m.predict([f"G{i}" for i in range(N_CAT)], "sysa")
    # fixed expected values (float32 arithmetic) guard against numerical drift across platforms and NumPy versions
    assert float(np.abs(r.values).sum()) == pytest.approx(GOLD_SUM, rel=1e-4)


def test_cli(tmp_path, capsys):
    path = tmp_path / "m.npz"
    np.savez(path, **make_open())
    assert cli_main(["info", "--model", str(path)]) == 0
    assert "systems: ['sysa']" in capsys.readouterr().out
    out = tmp_path / "o.csv"
    assert cli_main(["predict", "--model", str(path), "--line", "sysa", "--gene", "G0", "--gene", "G1", "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "output_gene,G0,G1" and len(lines) == NO + 1
    with pytest.raises(SystemExit) as e:
        cli_main(["--version"])
    assert e.value.code == 0
    assert "superslim" in capsys.readouterr().out


def test_rejects_other_files():
    with pytest.raises(ValueError):
        SuperSLIM({"catalog_b": _blob(["A"])})
