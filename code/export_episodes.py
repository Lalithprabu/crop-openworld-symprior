"""Writes every evaluation episode of the paper (support indices; queries = all images outside the support groups) to
release/benchmark/episodes.json, so other methods can be compared on exactly the same splits.
Indices refer to the rows of index_A.csv, index_B.csv, tomato_index.csv (field C) and index_source.csv.
Usage: python export_episodes.py
"""
import os, json
import numpy as np
import fewshot as F

OUT = os.path.join(F.ROOT, "release", "benchmark"); os.makedirs(OUT, exist_ok=True)
E = {"_readme": "support = row indices of the labelled photographs; every photograph whose group is not a support group is a query. "
                "Seeds and drawing rule: fewshot.draw(y, g, k, numpy.random.default_rng(seed))."}
S = np.load(os.path.join(F.OUT_CACHE, "fs_source_clip.npz")); ys, gs = S["y"], S["g"]
E["source_tau_selection"] = {f"k{k}": [F.draw(ys, gs, k, np.random.default_rng(77 + 100 * k + d))[0].tolist() for d in range(20)] for k in (1, 2, 5, 10)}
for ds in ("A", "B", "C"):
    Z = np.load(os.path.join(F.OUT_CACHE, f"fs_{ds}.npz")); y, g = Z["y"], Z["g"]
    E[f"fewshot_{ds}"] = {f"k{k}": [F.draw(y, g, k, np.random.default_rng(1000 * k + d))[0].tolist() for d in range(20)]
                          for k in F.KS if 0 < k <= F.max_k(y, g)}
Z = np.load(os.path.join(F.OUT_CACHE, "fs_B.npz")); y, g = Z["y"], Z["g"]; keep = np.where(y != 3)[0]
E["openset_B_yvmv_unknown"] = {f"k{k}": [keep[F.draw(y[keep], g[keep], k, np.random.default_rng(900 + 10 * k + d))[0]].tolist() for d in range(20)] for k in (1, 2)}
for name, (yy, gg, ncls, seed0) in {"openset_source_holdout": (ys, gs, 6, 500),
                                     "openset_C_holdout": (np.load(os.path.join(F.OUT_CACHE, "fs_C.npz"))["y"], np.load(os.path.join(F.OUT_CACHE, "fs_C.npz"))["g"], 8, 700)}.items():
    E[name] = {}
    for k in (1, 2, 5):
        for u in range(ncls):
            kp = np.where(yy != u)[0]
            E[name][f"k{k}|unknown{u}"] = [kp[F.draw(yy[kp], gg[kp], k, np.random.default_rng(seed0 + 10 * u + d))[0]].tolist() for d in range(10)]
json.dump(E, open(os.path.join(OUT, "episodes.json"), "w"))
print("episodes written:", {k: len(v) for k, v in E.items() if not k.startswith("_")})
