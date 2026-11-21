"""Runtime benchmark for a SuperSLIM model file: cold load, per-gene latency, batch throughput, peak memory, and numerical determinism.

    python benchmarks/bench_runtime.py --model model.npz [--json out.json] [--repeat 30]

Memory is measured in a fresh subprocess (peak working set / max RSS). Timings use time.perf_counter; the median over repeats is reported.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from superslim import SuperSLIM, __version__  # noqa: E402


def _peak_mb_subprocess(model: str) -> float:
    code = (
        "import sys, os; sys.path.insert(0, r'%s'); import numpy as np, superslim; m = superslim.SuperSLIM.load(r'%s'); g = m.genes()[:50]\n"
        "[m.predict(g, L) for L in m.lines]\n"
        "try:\n import psutil; p = psutil.Process().memory_info(); print(getattr(p, 'peak_wset', p.rss) / 2**20)\n"
        "except Exception:\n import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != 'darwin' else 1024**2))\n"
    ) % (os.path.join(os.path.dirname(HERE), "src"), model)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json")
    ap.add_argument("--repeat", type=int, default=30)
    a = ap.parse_args()
    res = {"superslim": __version__, "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(), "model_bytes": os.path.getsize(a.model)}
    loads = []
    for _ in range(5):
        t = time.perf_counter()
        m = SuperSLIM.load(a.model)
        loads.append((time.perf_counter() - t) * 1000)
    res["load_ms_median"] = statistics.median(loads)
    res["peak_memory_mb_after_all_systems"] = _peak_mb_subprocess(a.model)
    genes = m.genes()
    rng = np.random.default_rng(0)
    pick = list(rng.choice(genes, 1000, replace=False))
    per = {}
    for L in m.lines:
        m.predict(pick[:2], L)  # warm-up
        v = []
        for g in pick[: a.repeat]:
            t = time.perf_counter()
            m.predict([g], L)
            v.append((time.perf_counter() - t) * 1000)
        thr = {}
        for n in (10, 100, 1000):
            t = time.perf_counter()
            r = m.predict(pick[:n], L)
            dt = time.perf_counter() - t
            thr[str(n)] = {"genes_per_second": n / dt, "output_values_mb": r.values.nbytes / 2**20}
        per[L] = {"n_output_genes": int(len(m.output_genes(L))), "single_gene_ms_median": statistics.median(v), "single_gene_ms_p95": float(np.percentile(v, 95)), "batch": thr}
    res["systems"] = per
    # determinism: repeated calls and a fresh load give bit-identical output
    L0 = m.lines[0]
    x1 = m.predict(pick[:50], L0).values
    x2 = m.predict(pick[:50], L0).values
    x3 = SuperSLIM.load(a.model).predict(pick[:50], L0).values
    res["determinism"] = {"repeat_identical": bool(np.array_equal(x1, x2)), "reload_identical": bool(np.array_equal(x1, x3)), "finite": bool(np.isfinite(x1).all())}
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
