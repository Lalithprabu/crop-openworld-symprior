"""Builds the field-C index from a clone of the PlantDoc dataset (https://github.com/pratikkayal/PlantDoc-Dataset, CC BY 4.0).

Tomato classes only (8 classes; 'two spotted spider mites', with 2 images, is excluded). train/ and test/ are pooled, byte-identical
duplicates are removed (an MD5 group spanning two classes is dropped), and near-duplicates (64-bit difference hash, <= 6 bits) are
grouped transitively, exactly as for fields A and B; near-duplicate groups whose photographs carry different labels are dropped. Writes tomato_index.csv and tomato_images.zip (the images, renamed by index).
Usage: python prepare_tomato.py <PlantDoc folder> <output folder>
"""
import os, sys, hashlib, zipfile, csv
import numpy as np, cv2
from tomato_text import TOMATO_CLASSES, PLANTDOC_FOLDER


def dhash(bgr):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def main(root, out):
    os.makedirs(out, exist_ok=True)
    rows = []
    for lab, cname in enumerate(TOMATO_CLASSES):
        for split in ("train", "test"):
            d = os.path.join(root, split, PLANTDOC_FOLDER[cname])
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                p = os.path.join(d, f); rows.append([p, lab, split, hashlib.md5(open(p, "rb").read()).hexdigest()])
    by = {}
    for r in rows:
        by.setdefault(r[3], []).append(r)
    keep = [v[0] for v in by.values() if len({x[1] for x in v}) == 1]          # one copy per MD5; drop cross-class duplicates
    keep.sort(key=lambda r: (r[1], r[0]))
    ok, imgs = [], []
    for r in keep:
        im = cv2.imread(r[0])
        if im is not None:
            ok.append(r); imgs.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
    H = np.unpackbits(np.stack([dhash(x) for x in imgs]), axis=1).astype(np.int8)
    D = (H[:, None, :] != H[None, :, :]).sum(2); par = list(range(len(ok)))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in zip(*np.where(np.triu(D <= 6, 1))):
        par[f(i)] = f(j)
    g = np.unique([f(i) for i in range(len(ok))], return_inverse=True)[1]
    lab = np.array([r[1] for r in ok]); conflict = [gg for gg in np.unique(g) if len(set(lab[g == gg])) > 1]   # same photo, two labels
    sel = ~np.isin(g, conflict); ok = [r for r, s_ in zip(ok, sel) if s_]; g = np.unique(g[sel], return_inverse=True)[1]
    print("near-duplicate groups with conflicting labels dropped:", len(conflict), "images:", int((~sel).sum()))
    with open(os.path.join(out, "tomato_index.csv"), "w", newline="") as fh, zipfile.ZipFile(os.path.join(out, "tomato_images.zip"), "w", zipfile.ZIP_STORED) as z:
        w = csv.writer(fh); w.writerow(["img", "label", "class", "group", "plantdoc_split", "plantdoc_file", "md5"])
        for i, (r, gg) in enumerate(zip(ok, g)):
            name = f"{i:04d}{os.path.splitext(r[0])[1].lower()}"; z.write(r[0], name)
            w.writerow([name, r[1], TOMATO_CLASSES[r[1]], int(gg), r[2], os.path.relpath(r[0], root).replace(os.sep, "/"), r[3]])
    y = np.array([r[1] for r in ok])
    print("images", len(ok), "per class", np.bincount(y).tolist(), "groups", g.max() + 1,
          "groups per class", [len(np.unique(g[y == c])) for c in range(len(TOMATO_CLASSES))], "dropped", len(rows) - len(ok))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
