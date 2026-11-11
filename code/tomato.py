"""Field C: PlantDoc tomato (8 classes, web-collected field photographs), embedded with the same encoders as fields A and B.

Run on the laptop (needs the BioCLIP / BioCLIP-2 / DINOv2 weights):  python tomato.py embed
Input : <paper folder>/data_tomato/tomato_images.zip and tomato_index.csv   (made by prepare_tomato.py from PlantDoc)
Output: <paper folder>/cache/fs_C.npz  (labels, groups, image embeddings, class-description and bank embeddings; no images)
"""
import os, sys, csv, zipfile, time
import numpy as np
import fewshot as F                                      # same loading, encoders and text embedding as fields A and B
from tomato_text import TOMATO_CLASSES, TOMATO_PROMPTS, TOMATO_EXTRA_BANK

DATA = os.path.join(F.ROOT, "data_tomato")


@F.torch.no_grad()
def stage_embed():
    from fm_eval import encoders, embed
    import fm_text
    out = os.path.join(F.OUT_CACHE, "fs_C.npz")
    if os.path.exists(out):
        print("fs_C.npz already exists - delete it to recompute"); return
    img_dir = os.path.join(DATA, "images")
    if not os.path.isdir(img_dir):
        with zipfile.ZipFile(os.path.join(DATA, "tomato_images.zip")) as z:
            z.extractall(img_dir)
    rows = list(csv.DictReader(open(os.path.join(DATA, "tomato_index.csv"))))
    y = np.array([int(r["label"]) for r in rows]); g = np.array([int(r["group"]) for r in rows])
    t0 = time.time()
    I352, _ = F.load([(os.path.join(img_dir, r["img"]), int(r["label"])) for r in rows])
    print("images loaded:", len(y), "per class", np.bincount(y).tolist(), flush=True)
    encs, DEV = encoders()
    E = F.chunked_embed(lambda b: embed(encs, DEV, b), I352)
    print("image-only encoders done", sorted(E), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    clips, _ = fm_text.load_models(); C = {}
    missing = {"bioclip2", "bioclip"} - set(clips)
    if missing:
        raise SystemExit(f"STOP: could not load {sorted(missing)} - check the internet connection or the Hugging Face cache, then run again")
    for name, (m, pre, tok) in clips.items():
        C[f"clip_{name}"] = F.chunked_embed(lambda b, m=m, pre=pre, tok=tok: fm_text.encode(m, pre, tok, DEV, b)[0], I352)
        C[f"ctxt_{name}"] = F.clip_text(m, tok, DEV, [TOMATO_PROMPTS[c] for c in TOMATO_CLASSES])
        C[f"bank_{name}"] = np.concatenate([C[f"ctxt_{name}"], F.clip_text(m, tok, DEV, list(TOMATO_EXTRA_BANK.values()))])
        print(name, "done", C[f"clip_{name}"].shape, C[f"bank_{name}"].shape, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(out, y=y, g=g, **{f"emb_{k}": v for k, v in E.items()}, **C)
    print("saved", out, f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        stage_embed()
