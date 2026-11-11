"""Writes the public benchmark index for fields A and B and the source data (file, MD5, label, re-shot group), and checks it
against the cached labels and groups used in the paper. Run on the laptop, where the image folders are:
    python export_index.py <folder of ext_a> <folder of ext_b>
Output: <paper folder>/release/benchmark/index_A.csv, index_B.csv, index_source.csv
"""
import os, sys, csv, hashlib
import numpy as np, pandas as pd
import fewshot as F

OUT = os.path.join(F.ROOT, "release", "benchmark"); os.makedirs(OUT, exist_ok=True)


def main(root_a, root_b):
    for ds, root in (("A", root_a), ("B", root_b)):
        rows = F.scan(root, True); _, I256 = F.load(rows)
        g, _ = F.groups([p for p, _ in rows], I256); y = np.array([l for _, l in rows])
        Z = np.load(os.path.join(F.OUT_CACHE, f"fs_{ds}.npz"))
        ok_y, ok_g = np.array_equal(y, Z["y"]), np.array_equal(g, Z["g"])
        print(ds, len(rows), "images; labels match cache:", ok_y, "; groups match cache:", ok_g, flush=True)
        if not (ok_y and ok_g):
            raise SystemExit("STOP: the index does not reproduce the cached labels/groups - tell Claude")
        with open(os.path.join(OUT, f"index_{ds}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["idx", "file", "md5", "label", "class", "group"])
            for i, ((p, l), gg) in enumerate(zip(rows, g)):
                w.writerow([i, os.path.relpath(p, root).replace(os.sep, "/"), hashlib.md5(open(p, "rb").read()).hexdigest(), l, F.TCLS[l], int(gg)])
    idx = pd.read_csv(os.path.join(F.CACHE, "index.csv")); S = np.load(os.path.join(F.OUT_CACHE, "fs_source_clip.npz"))
    keep = [c for c in idx.columns if c.lower() in ("img_idx", "file", "path", "relpath", "md5", "label", "class")]
    src = idx[keep].copy(); src["group"] = S["g"]
    if not np.array_equal(idx.label.values, S["y"]):
        raise SystemExit("STOP: source labels do not match the cache - tell Claude")
    src.to_csv(os.path.join(OUT, "index_source.csv"), index=False); print("source", len(src), "images; columns", list(src.columns))
    print("saved to", OUT)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
