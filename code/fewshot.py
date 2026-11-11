"""Conference paper 2: SymPrior - symptom-description priors for few-shot okra disease recognition in new fields.

Method (SymPrior). In a vision-language foundation model (BioCLIP-2, BioCLIP) each class c has a text embedding t_c of its written
symptom description. After removing the modality gap (subtracting the mean image embedding of the unlabelled target photographs
and the mean text embedding of the description set, then re-normalising), the class prototype is the posterior mean direction of a
von Mises-Fisher model with conjugate prior centred on the description:
        p_c  propto  tau * t_c  +  sum_{i in S_c} x_i ,
so k = 0 gives zero-shot recognition from the description, and labelled photographs progressively take over. The single prior
strength tau is chosen on simulated few-shot episodes of Okra DiseaseNet (the source data) - never on the target fields.

How many labelled leaves does a new field need?

Label-efficient adaptation of okra leaf-disease classifiers to two independent fields (public datasets A and B), including a
disease never seen in the source data (yellow vein mosaic virus, YVMV, in dataset B).

Leakage control (as in the journal study): exact duplicates removed (MD5); near-duplicates (dHash <= 6 bits) and photographs
taken within 5 s of each other (EXIF, when present) are grouped, and support (labelled) and query (test) images never share a group.

Stages (each saves its output, so a stage can be re-run alone):
  embed     load + deduplicate + group the target images; train the source model on all 293 Okra DiseaseNet photographs;
            extract features with ImageNet MobileNetV3, the source-tuned MobileNetV3, BioCLIP and DINOv2 -> cache/fs_<ds>.npz
  probe     k-shot linear probe and nearest-centroid classifiers, k in KS, N_DRAWS random support sets -> results/fs_probe.json
  finetune  k-shot full fine-tuning of MobileNetV3 from ImageNet weights vs from the source model -> results/fs_finetune.json

Usage (from this folder):  python fewshot.py embed <folder of ext_a> <folder of ext_b>
        python fewshot.py probe
        python fewshot.py finetune
Set OKRA_QUICK=1 for a smoke test (few draws, 2 epochs).
"""
import os, sys, json, hashlib, time
import numpy as np, cv2, torch, torch.nn as nn
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
# This paper lives in its own folder. The journal code and the Okra DiseaseNet cache are only READ, never written:
# all outputs go to <paper folder>/cache and <paper folder>/results.
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
JSRC = os.environ.get("OKRA_JOURNAL_SRC", r"A:\PhD\Novelty\Revision_Submission\okra_revision_code_and_results\okra_revision\src")
sys.path.insert(0, JSRC)
from common import CACHE, CLASSES, set_seed, get_device          # CACHE = journal cache (read-only here)
import shortcut as SC
from tomato_text import TOMATO_CLASSES, TOMATO_EXTRA_BANK   # second crop (field C)
OUT_CACHE, OUT_RES = os.path.join(ROOT, "cache"), os.path.join(ROOT, "results")
os.makedirs(OUT_CACHE, exist_ok=True); os.makedirs(OUT_RES, exist_ok=True)


def save_json(obj, name):
    json.dump(obj, open(os.path.join(OUT_RES, name), "w"), indent=1)

QUICK = os.environ.get("OKRA_QUICK") == "1"
# Thermal safety for laptops: fewer CPU threads, small chunks and short pauses (OKRA_COOL seconds per chunk; 0 = full speed).
COOL = float(os.environ.get("OKRA_COOL", "1.0"))
torch.set_num_threads(int(os.environ.get("OKRA_THREADS", "2"))); cv2.setNumThreads(1)


def rest(f=1.0):
    if COOL > 0:
        time.sleep(COOL * f)


def chunked_embed(fn, imgs, n=64):
    """Run an embedding function on small chunks with pauses; concatenates dict or array outputs."""
    parts = []
    for i in range(0, len(imgs), n):
        parts.append(fn(imgs[i:i + n])); rest()
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


OKRA = ("okra", "ladies", "lady", "bhindi")
# target label space: 0 healthy, 1 Cercospora leaf spot, 2 leaf curl, 3 yellow vein mosaic (new disease)
TCLS = ["Healthy", "Cercospora leaf spot", "Leaf curl", "Yellow vein mosaic"]
KEYS = [("ospora", 1), ("curl", 2), ("yellow vein", 3), ("yvmv", 3), ("healt", 0)]
SRC_OF = {0: 0, 1: 2, 2: 4}            # target class -> Okra DiseaseNet class index (H=0, CLS=2, LCV=4); YVMV has none
KS = [0, 1, 2, 5, 10, 20, 40]
YVMV_PROMPTS = ["a photo of an okra leaf with a bright yellow network of veins and mosaic yellowing",
                "yellow vein mosaic virus on okra: vein clearing, yellow veins forming a net, chlorotic mosaic leaf"]
TAUS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
# Open-vocabulary description bank used to NAME an unknown disease (fixed before any result was seen). The first 7 entries are
# the 6 Okra DiseaseNet classes (fm_text.PROMPTS) and YVMV; the rest are common okra disorders that act as plausible distractors.
EXTRA_BANK = {
    "Powdery mildew": ["a photo of an okra leaf covered with white powdery patches",
                       "powdery mildew on okra: white to grey powdery fungal growth on the leaf surface"],
    "Enation leaf curl": ["a photo of an okra leaf curled upwards with small outgrowths on the underside veins",
                          "okra enation leaf curl virus: thickened veins with enations and twisted, curled leaves"],
    "Bacterial leaf spot": ["a photo of an okra leaf with small water-soaked angular spots",
                            "bacterial leaf spot on okra: dark angular water-soaked lesions limited by veins"],
    "Anthracnose": ["a photo of an okra leaf with sunken dark brown lesions",
                    "anthracnose on okra: irregular dark sunken necrotic lesions on the leaf"],
    "Fusarium wilt": ["a photo of a yellowing wilting okra leaf",
                      "fusarium wilt on okra: progressive yellowing and wilting of the whole leaf"],
    "Jassid hopper burn": ["a photo of an okra leaf with yellowing and browning margins curling downwards",
                           "leafhopper (jassid) damage on okra: hopper burn, yellow then brown leaf edges, downward cupping"],
    "Spider mite damage": ["a photo of an okra leaf with fine pale stippling and webbing",
                           "red spider mite damage on okra: tiny yellow speckles, bronzing and fine webs"],
    "Nitrogen deficiency": ["a photo of a uniformly pale yellow-green okra leaf",
                            "nitrogen deficiency in okra: general chlorosis of the whole leaf without spots"],
}
N_DRAWS = 3 if QUICK else 20
FT_KS = [1, 2, 5, 10, 20]
FT_DRAWS = 1 if QUICK else 5
FT_EPOCHS = 2 if QUICK else 30


# ------------------------------------------------------------------ data
def scan(root, need_okra):
    rows, seen = [], set()
    for dp, dn, fn in os.walk(root):
        low = dp.lower().replace("\\", "/")
        if "augment" in low or (need_okra and not any(k in low for k in OKRA)):
            continue
        leaf = low.split("/")[-1]
        lab = next((c for k, c in KEYS if k in leaf), None)
        if lab is None:
            continue
        for f in sorted(fn):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p = os.path.join(dp, f); h = hashlib.md5(open(p, "rb").read()).hexdigest()
                if h not in seen:
                    seen.add(h); rows.append((p, lab))
    return rows


def dhash(bgr):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def exif_time(p):
    try:
        v = Image.open(p).getexif().get(306) or Image.open(p)._getexif().get(36867)
        return time.mktime(time.strptime(v, "%Y:%m:%d %H:%M:%S"))
    except Exception:
        return None


def groups(paths, I256):
    n = len(paths); par = list(range(n))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    H = np.stack([dhash(x) for x in I256])
    bits = np.unpackbits(H, axis=1).astype(np.int8)
    D = (bits[:, None, :] != bits[None, :, :]).sum(2)
    for i, j in zip(*np.where(np.triu(D <= 6, 1))):
        par[f(i)] = f(j)
    t = [exif_time(p) for p in paths]
    idx = [i for i in range(n) if t[i] is not None]
    idx.sort(key=lambda i: t[i])
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] <= 5:
            par[f(a)] = f(b)
    g = np.array([f(i) for i in range(n)])
    return np.unique(g, return_inverse=True)[1], int(sum(x is not None for x in t))


def load(rows):
    I352, I256 = [], []
    for n_, (p, _) in enumerate(rows):
        if n_ % 50 == 49:
            rest(0.5); print("  loaded", n_ + 1, "of", len(rows), flush=True)
        im = Image.open(p); im.draft("RGB", (1536, 1536))                       # fast, low-heat JPEG decode at reduced size
        im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((768, 768), Image.LANCZOS)
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        I352.append(cv2.resize(bgr, (352, 352), interpolation=cv2.INTER_AREA))
        I256.append(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA))
    return np.stack(I352), np.stack(I256)


# ------------------------------------------------------------------ features
def source_model():
    """MobileNetV3-Large trained on all 293 Okra DiseaseNet photographs (journal recipe, full image, seed 0)."""
    import pandas as pd
    p = os.path.join(OUT_CACHE, "fs_source_mnv3.pt")
    idx = pd.read_csv(os.path.join(CACHE, "index.csv")); ii = idx.img_idx.values; y = idx.label.values
    m = SC.build("mobilenetv3", len(CLASSES))
    if os.path.exists(p):
        m.load_state_dict(torch.load(p, map_location="cpu")); return m.to(get_device()).eval()
    imgs = np.load(os.path.join(CACHE, "imgs352.npy"))[ii]; M = np.ones(imgs.shape[:3], bool)
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=0)
    if QUICK:
        SC.EPOCHS = 2
    m, ep = SC.train(imgs, M, y, tr, va, "full", len(CLASSES), seed=0)
    torch.save(m.state_dict(), p); print("source model trained,", ep, "epochs"); return m


@torch.no_grad()
def penultimate(m, I352):
    DEV = next(m.parameters()).device; out = []
    dl = torch.utils.data.DataLoader(SC.MaskDS(I352, np.ones(I352.shape[:3], bool), np.zeros(len(I352), int), False, "full"), batch_size=32)
    for xb, _ in dl:
        out.append(m.forward_head(m.forward_features(xb.to(DEV)), pre_logits=True).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def clip_text(m, tok, DEV, prompts):
    T = []
    for ps in prompts:
        z = torch.nn.functional.normalize(m.encode_text(tok(ps).to(DEV)).float(), dim=-1).mean(0)
        T.append(torch.nn.functional.normalize(z, dim=0).cpu().numpy())
    return np.stack(T)


def target_prompts():
    from fm_text import PROMPTS
    return [PROMPTS["Healthy"], PROMPTS["CLS"], PROMPTS["LCV"], YVMV_PROMPTS]       # order = TCLS


def stage_embed(root_a, root_b):
    from fm_eval import encoders, embed
    import fm_text, pandas as pd
    encs, DEV = encoders()
    clips, _ = fm_text.load_models()
    src_clip = os.path.join(OUT_CACHE, "fs_source_clip.npz")
    todo = [ds for ds in ("A", "B") if not os.path.exists(os.path.join(OUT_CACHE, f"fs_{ds}.npz"))]
    if os.path.exists(src_clip) and not todo:
        print("embed: all outputs exist - skipped (delete the cache folder to recompute)"); return
    # source episodes for choosing tau: Okra DiseaseNet images, labels, burst-aware groups, and class texts
    idx = pd.read_csv(os.path.join(CACHE, "index.csv")); ii = idx.img_idx.values
    gsrc = idx.group.values
    ft5 = os.path.join(CACHE, "folds_t5b.csv")
    if os.path.exists(ft5):
        f5 = pd.read_csv(ft5); gsrc = idx.img_idx.map(dict(zip(f5.img_idx, f5.group_t5))).values
    Isrc = np.load(os.path.join(CACHE, "imgs352.npy"))[ii]; srcsave = {"y": idx.label.values, "g": np.unique(gsrc, return_inverse=True)[1]}
    for name, (m, pre, tok) in (clips.items() if not os.path.exists(src_clip) else []):     # resume: skip if already saved
        Z = chunked_embed(lambda b, m=m, pre=pre, tok=tok: fm_text.encode(m, pre, tok, DEV, b)[0], Isrc)
        srcsave[f"img_{name}"] = Z
        srcsave[f"txt_{name}"] = clip_text(m, tok, DEV, [fm_text.PROMPTS[c] for c in fm_text.PROMPTS] + [YVMV_PROMPTS])   # 6 source + YVMV
        srcsave[f"bank_{name}"] = np.concatenate([srcsave[f"txt_{name}"], clip_text(m, tok, DEV, list(EXTRA_BANK.values()))])
    if not os.path.exists(src_clip):
        np.savez_compressed(src_clip, **srcsave)
    src = source_model()
    for ds, root, need in (("A", root_a, True), ("B", root_b, True)):
        if ds not in todo:
            print(ds, "already embedded - skipped"); continue
        rows = scan(root, need)
        if not rows:
            print("no images found for", ds, root); continue
        I352, I256 = load(rows); y = np.array([l for _, l in rows])
        g, n_exif = groups([p for p, _ in rows], I256)
        print(ds, "images loaded; extracting features", flush=True)
        E = chunked_embed(lambda b: embed(encs, DEV, b), I352)
        E["source"] = chunked_embed(lambda b: penultimate(src, b), I352)
        C = {}
        for name, (m, pre, tok) in clips.items():
            Z = chunked_embed(lambda b, m=m, pre=pre, tok=tok: fm_text.encode(m, pre, tok, DEV, b)[0], I352)
            C[f"clip_{name}"] = Z; C[f"ctxt_{name}"] = clip_text(m, tok, DEV, target_prompts())
        np.savez_compressed(os.path.join(OUT_CACHE, f"fs_{ds}.npz"), y=y, g=g, I352=I352, **{f"emb_{k}": v for k, v in E.items()}, **C)
        print(ds, "images", len(y), "per class", np.bincount(y, minlength=4).tolist(), "groups", g.max() + 1, "with EXIF", n_exif)


# ------------------------------------------------------------------ k-shot sampling
def draw(y, g, k, rng):
    """k support photographs per class, taken from distinct groups; query = every photograph outside the support groups."""
    sup = []
    for c in np.unique(y):
        gs = np.unique(g[y == c]); rng.shuffle(gs)
        for gg in gs[:k]:
            sup.append(rng.choice(np.where((g == gg) & (y == c))[0]))
    sup = np.array(sup); qry = np.where(~np.isin(g, g[sup]))[0]
    return sup, qry


def scores(y, p):
    s = {"bal_acc": float(100 * balanced_accuracy_score(y, p)), "macro_f1": float(100 * f1_score(y, p, average="macro"))}
    if (y == 3).any() and y.max() == 3:                  # field B only: recall of YVMV, the disease never seen in the source data
        s["yvmv_recall"] = float(100 * (p[y == 3] == 3).mean())
    return s


def summarise(res):
    return {key: {m: [float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))] for m in v[0]} | {"n": len(v)}
            for key, v in res.items()}


def max_k(y, g):
    return min(len(np.unique(g[y == c])) for c in np.unique(y)) // 2      # keep at least half of the groups for testing


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def symprior(Xs, ys, Xq, T, cls, tau, mu_img, mu_txt, gap=True):
    """Posterior-mean prototypes (vMF, conjugate prior centred on the description). Returns predicted classes for Xq."""
    if gap:
        Xs, Xq, T = unit(Xs - mu_img), unit(Xq - mu_img), unit(T - mu_txt)
    P = np.stack([unit(tau * T[j] + Xs[ys == c].sum(0)) if (ys == c).any() or tau > 0 else T[j] for j, c in enumerate(cls)])
    return cls[(Xq @ P.T).argmax(1)]


def tip_adapter(Xs, ys, Xq, T, cls, alpha=1.0, beta=5.5):
    """Training-free Tip-Adapter (Zhang et al., ECCV 2022) with its default alpha, beta."""
    zs = 100 * Xq @ T.T
    if len(ys) == 0:
        return cls[zs.argmax(1)]
    L = (ys[:, None] == cls[None, :]).astype(float)
    return cls[(zs + alpha * np.exp(-beta * (1 - Xq @ Xs.T)) @ L).argmax(1)]


LAMBDAS = [round(0.1 * i, 1) for i in range(0, 11)]           # the full range [0, 1] of Goswami et al.


def align_mix(Xs, ys, Xq, T, cls, lam):
    """Align+Mix (Goswami et al., 2026), training-free: project onto the text subspace (truncated SVD of the text prototypes,
    99.9% of the variance), then mix image and text prototypes, lam * P mu_img + (1 - lam) * mu_text; cosine nearest prototype."""
    U, sv, _ = np.linalg.svd(unit(T).T, full_matrices=False)
    r = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.999) + 1); P = U[:, :r] @ U[:, :r].T
    Xs, Xq, Tn = unit(Xs), unit(Xq), unit(T)
    M = np.stack([unit(lam * (P @ unit(Xs[ys == c].mean(0))) + (1 - lam) * Tn[j]) for j, c in enumerate(cls)])
    return cls[(unit(Xq @ P) @ M.T).argmax(1)]


def choose_lambda(model):
    """Align+Mix weight chosen exactly like tau: on few-shot episodes of the SOURCE data, never on target labels."""
    S = np.load(os.path.join(OUT_CACHE, "fs_source_clip.npz")); X, y, g = S[f"img_{model}"], S["y"], S["g"]
    T = S[f"txt_{model}"][:len(CLASSES)]; cls = np.arange(len(CLASSES)); score = {l: [] for l in LAMBDAS}
    for k in (1, 2, 5, 10):
        for d in range(5 if QUICK else 20):
            rng = np.random.default_rng(77 + 100 * k + d); sup, qry = draw(y, g, k, rng)
            for l in LAMBDAS:
                score[l].append(balanced_accuracy_score(y[qry], align_mix(X[sup], y[sup], X[qry], T, cls, l)))
    return max(LAMBDAS, key=lambda a: np.mean(score[a])), {str(a): float(100 * np.mean(v)) for a, v in score.items()}


def txt_mean(S, mdl):
    """Text centre for the gap correction: mean of all 15 description-bank entries (fixed before any result, the same for every field)."""
    return S[f"bank_{mdl}"].mean(0)


def choose_tau(model):
    """Prior strength chosen on few-shot episodes of the SOURCE data (Okra DiseaseNet, 6 classes); target labels never used."""
    S = np.load(os.path.join(OUT_CACHE, "fs_source_clip.npz")); X, y, g = S[f"img_{model}"], S["y"], S["g"]
    T = S[f"txt_{model}"]; mu_txt = txt_mean(S, model); T = T[:len(CLASSES)]; mu_img = X.mean(0); cls = np.arange(len(CLASSES))
    score = {tau: [] for tau in TAUS}
    for k in (1, 2, 5, 10):
        for d in range(5 if QUICK else 20):
            rng = np.random.default_rng(77 + 100 * k + d); sup, qry = draw(y, g, k, rng)
            for tau in TAUS:
                score[tau].append(balanced_accuracy_score(y[qry], symprior(X[sup], y[sup], X[qry], T, cls, tau, mu_img, mu_txt)))
    best = max(TAUS, key=lambda a: np.mean(score[a]))
    return best, {str(a): float(100 * np.mean(v)) for a, v in score.items()}


RAW = {}


def stage_probe():
    out = {}
    taus = {}
    if os.path.exists(os.path.join(OUT_CACHE, "fs_source_clip.npz")):
        for mdl in [k[4:] for k in np.load(os.path.join(OUT_CACHE, "fs_source_clip.npz")).files if k.startswith("img_")]:
            taus[mdl] = choose_tau(mdl); print("tau for", mdl, "=", taus[mdl][0], flush=True)
    lams = {m: choose_lambda(m) for m in taus}
    out["_tau"] = {m: {"chosen": v[0], "source_episode_bal_acc": v[1]} for m, v in taus.items()}
    out["_lambda"] = {m: {"chosen": v[0], "source_episode_bal_acc": v[1]} for m, v in lams.items()}
    for ds in ("A", "B", "C"):                          # C = PlantDoc tomato (second crop), embedded by tomato.py
        fp = os.path.join(OUT_CACHE, f"fs_{ds}.npz")
        if not os.path.exists(fp):
            continue
        Z = np.load(fp); y, g = Z["y"], Z["g"]; feats = [k[4:] for k in Z.files if k.startswith("emb_")]
        res = {}
        clipm = [k[5:] for k in Z.files if k.startswith("clip_")]; cls = np.unique(y)
        for k in [k for k in KS if k <= max_k(y, g)]:
            for d in range(1 if k == 0 else N_DRAWS):
                rng = np.random.default_rng(1000 * k + d); sup, qry = draw(y, g, k, rng) if k > 0 else (np.array([], int), np.arange(len(y)))
                for mdl in clipm:                              # vision-language methods (SymPrior and baselines)
                    X = Z[f"clip_{mdl}"]; T = Z[f"ctxt_{mdl}"][cls]
                    S = np.load(os.path.join(OUT_CACHE, "fs_source_clip.npz"))
                    mu_txt = Z[f"bank_{mdl}"].mean(0) if f"bank_{mdl}" in Z.files else txt_mean(S, mdl)   # each crop: mean of its own bank
                    mu_img = X.mean(0)                          # transductive but label-free: mean of the unlabelled target photographs
                    tau = taus.get(mdl, (4,))[0]
                    mu_src = S[f"img_{mdl}"].mean(0)            # inductive variant: image mean of the SOURCE data only
                    P = {"zeroshot": cls[(X[qry] @ T.T).argmax(1)],
                         "symprior": symprior(X[sup], y[sup], X[qry], T, cls, tau, mu_img, mu_txt),
                         "symprior_inductive": symprior(X[sup], y[sup], X[qry], T, cls, tau, mu_src, mu_txt),
                         "symprior_nogap": symprior(X[sup], y[sup], X[qry], T, cls, tau, mu_img, mu_txt, gap=False),
                         "tipadapter": tip_adapter(X[sup], y[sup], X[qry], T, cls)}
                    if k > 0:
                        Xg, Tg = unit(X - mu_img), unit(T - mu_txt)          # Tip-Adapter on gap-corrected embeddings
                        P["tipadapter_gap"] = tip_adapter(Xg[sup], y[sup], Xg[qry], Tg, cls)
                        P["tipadapter_gap_tuned"] = max((tip_adapter(Xg[sup], y[sup], Xg[qry], Tg, cls, a, b) for a in (0.5, 1, 2, 5, 10)
                                                         for b in (1, 3, 5.5, 10)), key=lambda p: balanced_accuracy_score(y[qry], p))  # tuned on the test photos: optimistic upper bound
                        P["alignmix"] = align_mix(X[sup], y[sup], X[qry], T, cls, lams[mdl][0])
                        P["alignmix_tuned"] = max((align_mix(X[sup], y[sup], X[qry], T, cls, l) for l in LAMBDAS),
                                                  key=lambda p: balanced_accuracy_score(y[qry], p))    # tuned on the test photos: upper bound
                        P["centroid"] = symprior(X[sup], y[sup], X[qry], T, cls, 0.0, mu_img, mu_txt, gap=False)
                        P["probe"] = LogisticRegression(C=1.0, max_iter=5000).fit(X[sup], y[sup]).predict(X[qry])
                    for meth, p in P.items():
                        res.setdefault(f"clip_{mdl}|{meth}|{k}", []).append(scores(y[qry], p))
                if k == 0:
                    continue
                for fn in feats:
                    X = Z[f"emb_{fn}"]; X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                    C = np.stack([X[sup][y[sup] == c].mean(0) for c in np.unique(y)])          # nearest class centroid (cosine)
                    pc = np.unique(y)[(X[qry] @ C.T).argmax(1)]
                    lr = LogisticRegression(C=1.0, max_iter=5000).fit(X[sup], y[sup])            # linear probe, fixed C (no validation set at k=1)
                    for meth, p in (("centroid", pc), ("probe", lr.predict(X[qry]))):
                        res.setdefault(f"{fn}|{meth}|{k}", []).append(scores(y[qry], p))
            print(ds, "k =", k, "done", flush=True)
        out[ds] = summarise(res); RAW[ds] = res
        out[ds]["_info"] = {"n_images": int(len(y)), "per_class": np.bincount(y, minlength=4).tolist(), "n_groups": int(g.max() + 1),
                            "classes": [(TOMATO_CLASSES if ds == "C" else TCLS)[c] for c in np.unique(y)], "max_k": int(max_k(y, g))}
    fp = os.path.join(OUT_CACHE, "fs_C.npz")
    if os.path.exists(fp):                              # field C: how good was the tau transferred from okra? (all tau values)
        Z = np.load(fp); y, g = Z["y"], Z["g"]; cls = np.unique(y); X = Z["clip_bioclip2"]; T = Z["ctxt_bioclip2"]
        mu_t, sens = Z["bank_bioclip2"].mean(0), {}
        for k in (1, 2, 5):
            for tau in TAUS:
                v = [balanced_accuracy_score(y[q], symprior(X[s_], y[s_], X[q], T, cls, tau, X.mean(0), mu_t))
                     for s_, q in (draw(y, g, k, np.random.default_rng(1000 * k + d)) for d in range(N_DRAWS))]
                sens[f"k{k}|tau{tau}"] = float(100 * np.mean(v))
        out["_tau_sensitivity_C"] = sens
    save_json(out, "fs_probe.json"); save_json(RAW, "fs_probe_raw.json")       # raw: one record per episode (paired tests)
    print("saved", os.path.join(OUT_RES, "fs_probe.json"))


# ------------------------------------------------------------------ fine-tuning
def finetune(I352, y, sup, init, seed):
    DEV = get_device(); set_seed(seed); cls = np.unique(y); n = len(cls); ymap = {c: i for i, c in enumerate(cls)}
    m = SC.build("mobilenetv3", len(CLASSES))
    if init == "source":
        m.load_state_dict(source_model().state_dict())
    m.reset_classifier(n)
    if init == "source":                       # start the new head from the source classifier rows where the class exists
        src = SC.build("mobilenetv3", len(CLASSES)); src.load_state_dict(torch.load(os.path.join(OUT_CACHE, "fs_source_mnv3.pt"), map_location="cpu"))
        with torch.no_grad():
            for c in cls:
                if c in SRC_OF:
                    m.classifier.weight[ymap[c]] = src.classifier.weight[SRC_OF[c]]; m.classifier.bias[ymap[c]] = src.classifier.bias[SRC_OF[c]]
    m = m.to(DEV); yy = np.array([ymap[c] for c in y[sup]])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS); lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    ds = SC.MaskDS(I352[sup], np.ones((len(sup),) + I352.shape[1:3], bool), yy, True, "full", seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=min(16, len(sup)), shuffle=True, drop_last=len(sup) > 16)
    for ep in range(FT_EPOCHS):                # fixed schedule: with k labelled images per class there is no validation set
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); lossf(m(xb.to(DEV)), yb.to(DEV)).backward(); opt.step(); rest(0.15)
        sch.step()
    m.eval(); return m, cls


def stage_finetune():
    out = {}
    for ds in ("A", "B"):
        fp = os.path.join(OUT_CACHE, f"fs_{ds}.npz")
        if not os.path.exists(fp):
            continue
        Z = np.load(fp); y, g, I352 = Z["y"], Z["g"], Z["I352"]; res = {}
        for k in [k for k in FT_KS if k <= max_k(y, g)]:
            for d in range(FT_DRAWS):
                rng = np.random.default_rng(1000 * k + d); sup, qry = draw(y, g, k, rng)    # same support sets as the probes
                for init in ("imagenet", "source"):
                    m, cls = finetune(I352, y, sup, init, seed=d)
                    P = SC.predict(m, I352[qry], np.ones((len(qry),) + I352.shape[1:3], bool), y[qry], "full")
                    res.setdefault(f"mnv3_ft_{init}|ft|{k}", []).append(scores(y[qry], cls[P.argmax(1)]))
                    del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    rest(3)
                print(ds, "k =", k, "draw", d, {kk: round(v[-1]["bal_acc"], 1) for kk, v in res.items() if kk.endswith(f"|{k}")}, flush=True)
        out[ds] = summarise(res); RAW[ds] = res
    save_json(out, "fs_finetune.json"); save_json(RAW, "fs_finetune_raw.json"); print("saved", os.path.join(OUT_RES, "fs_finetune.json"))


# ------------------------------------------------------------------ open world: detect -> name -> learn
def bank_names():
    return ["Healthy", "Alternaria leaf spot", "Cercospora leaf spot", "Downy mildew", "Leaf curl", "Phyllosticta leaf spot",
            "Yellow vein mosaic"] + list(EXTRA_BANK)


def open_scores(Xs, ys, Xq, known, T=None, tau=0.0, mu_img=None, mu_txt=None):
    """Unknown-ness scores for the queries (higher = more likely unknown) and the closed-set prediction."""
    out = {}
    if T is not None:                                    # SymPrior prototypes (gap-corrected), score = 1 - max cosine
        Xs_, Xq_, T_ = unit(Xs - mu_img), unit(Xq - mu_img), unit(T - mu_txt)
        P = np.stack([unit(tau * T_[j] + Xs_[ys == c].sum(0)) for j, c in enumerate(known)])
        s = Xq_ @ P.T; out["symprior"] = (1 - s.max(1), known[s.argmax(1)])
    C = np.stack([unit(Xs[ys == c].mean(0)) for c in known]); s = Xq @ C.T
    out["centroid"] = (1 - s.max(1), known[s.argmax(1)])
    out["knn"] = (1 - (Xq @ Xs.T).max(1), known[(Xq @ C.T).argmax(1)])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(Xs, ys); pr = lr.predict_proba(Xq)
    out["msp"] = (1 - pr.max(1), lr.classes_[pr.argmax(1)])
    return out


def name_unknown(Xu, bank, known_names, mu_img, mu_txt, names=None):
    """Name the unknown photographs from the description bank, excluding the known classes. Returns per-image names and the
    name of the mean embedding (all flagged photographs together)."""
    names = names or bank_names(); allowed = np.array([n not in known_names for n in names])
    B = unit(bank - mu_txt); Xu = unit(Xu - mu_img); s = Xu @ B.T; s[:, ~allowed] = -9
    sm = unit(Xu.mean(0)) @ B.T; sm[~allowed] = -9
    return [names[i] for i in s.argmax(1)], names[int(sm.argmax())], [names[i] for i in np.argsort(-sm)[:3]]


def fpr95(isu, score):
    """Share of known-class queries flagged as unknown when 95% of the unknown queries are flagged (lower is better)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(isu, score); return float(100 * fpr[np.searchsorted(tpr, 0.95)])


def stage_openset():
    from sklearn.metrics import roc_auc_score
    out = {}; S = np.load(os.path.join(OUT_CACHE, "fs_source_clip.npz")); clipm = [k[4:] for k in S.files if k.startswith("img_")]
    taus = json.load(open(os.path.join(OUT_RES, "fs_probe.json")))["_tau"] if os.path.exists(os.path.join(OUT_RES, "fs_probe.json")) else {}
    names = bank_names(); src_names = names[:len(CLASSES)]
    # (1) Okra DiseaseNet, each class held out in turn as the unknown disease (group-aware support/query)
    for mdl in clipm:
        X, y, g, bank = S[f"img_{mdl}"], S["y"], S["g"], S[f"bank_{mdl}"]; mu_txt = txt_mean(S, mdl); mu_img = X.mean(0)
        tau = taus.get(mdl, {}).get("chosen", 4)
        for k in [k for k in (1, 2, 5) if k <= max_k(y, g)]:
            rec = {}
            for u in range(len(CLASSES)):
                known = np.array([c for c in range(len(CLASSES)) if c != u]); keep = y != u
                for d in range(3 if QUICK else 10):
                    rng = np.random.default_rng(500 + 10 * u + d)
                    sup_k, _ = draw(y[keep], g[keep], k, rng); sup = np.where(keep)[0][sup_k]
                    qry = np.where(~np.isin(g, g[sup]))[0]; isu = (y[qry] == u)
                    sc = open_scores(X[sup], y[sup], X[qry], known, bank[known], tau, mu_img, mu_txt)
                    for meth, (score, _) in sc.items():
                        rec.setdefault(f"auroc_{meth}", []).append(100 * roc_auc_score(isu, score))
                        rec.setdefault(f"fpr95_{meth}", []).append(fpr95(isu, score))
                    per, grp, top3 = name_unknown(X[qry][isu], bank, [src_names[c] for c in known], mu_img, mu_txt)
                    rec.setdefault("name_img_top1", []).append(100 * np.mean([p == src_names[u] for p in per]))
                    rec.setdefault("name_group_top1", []).append(100.0 * (grp == src_names[u]))
                    rec.setdefault("name_group_top3", []).append(100.0 * (src_names[u] in top3))
                    rec.setdefault(f"per_class_name_group_{src_names[u]}", []).append(100.0 * (grp == src_names[u]))
            out[f"source_loco|{mdl}|k{k}"] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in rec.items()}
            RAW[f"source_loco|{mdl}|k{k}"] = {m: [float(x) for x in v] for m, v in rec.items()}
    # (2) new field B: known = healthy, Cercospora, leaf curl (k labelled photographs each); unknown = yellow vein mosaic
    fp = os.path.join(OUT_CACHE, "fs_B.npz")
    if os.path.exists(fp):
        Z = np.load(fp); y, g = Z["y"], Z["g"]; known = np.array([0, 1, 2]); tn = {0: "Healthy", 1: "Cercospora leaf spot", 2: "Leaf curl", 3: "Yellow vein mosaic"}
        feats = [k[4:] for k in Z.files if k.startswith("emb_")] + [f"clip:{m}" for m in clipm]
        for k in [k for k in (1, 2, 5) if k <= max_k(y[y != 3], g[y != 3])]:
            rec = {}
            for d in range(3 if QUICK else 20):
                rng = np.random.default_rng(900 + 10 * k + d); keep = y != 3
                sup_k, _ = draw(y[keep], g[keep], k, rng); sup = np.where(keep)[0][sup_k]
                qry = np.where(~np.isin(g, g[sup]))[0]; isu = (y[qry] == 3)
                for fn in feats:
                    if fn.startswith("clip:"):
                        mdl = fn[5:]; X = Z[f"clip_{mdl}"]; T = Z[f"ctxt_{mdl}"][known]; mu_img = X.mean(0); mu_txt = txt_mean(S, mdl)
                        sc = open_scores(X[sup], y[sup], X[qry], known, T, taus.get(mdl, {}).get("chosen", 4), mu_img, mu_txt)
                        per, grp, top3 = name_unknown(X[qry][isu], S[f"bank_{mdl}"], [tn[c] for c in known], mu_img, mu_txt)
                        rec.setdefault(f"{fn}|name_img_top1", []).append(100 * np.mean([p == "Yellow vein mosaic" for p in per]))
                        rec.setdefault(f"{fn}|name_group_top1", []).append(100.0 * (grp == "Yellow vein mosaic"))
                        rec.setdefault(f"{fn}|name_group_top3", []).append(100.0 * ("Yellow vein mosaic" in top3))
                        for fr in (0.1, 0.2, 0.4):                    # realistic naming: pool the queries with the highest unknown score
                            idx = np.argsort(-sc["symprior"][0])[:int(round(fr * len(qry)))]
                            _, gf, t3f = name_unknown(X[qry][idx], S[f"bank_{mdl}"], [tn[c] for c in known], mu_img, mu_txt)
                            rec.setdefault(f"{fn}|flag{int(100*fr)}_precision", []).append(100 * isu[idx].mean())
                            rec.setdefault(f"{fn}|flag{int(100*fr)}_name_top1", []).append(100.0 * (gf == "Yellow vein mosaic"))
                            rec.setdefault(f"{fn}|flag{int(100*fr)}_name_top3", []).append(100.0 * ("Yellow vein mosaic" in t3f))
                        mu_noU = X[y != 3].mean(0)                    # sensitivity: image centre without the unknown class in the pool
                        sc2 = open_scores(X[sup], y[sup], X[qry], known, T, taus.get(mdl, {}).get("chosen", 4), mu_noU, mu_txt)
                        rec.setdefault(f"{fn}|auroc_symprior_poolnoU", []).append(100 * roc_auc_score(isu, sc2["symprior"][0]))
                        _, g2, t32 = name_unknown(X[qry][isu], S[f"bank_{mdl}"], [tn[c] for c in known], mu_noU, mu_txt)
                        rec.setdefault(f"{fn}|name_group_top1_poolnoU", []).append(100.0 * (g2 == "Yellow vein mosaic"))
                    else:
                        X = unit(Z[f"emb_{fn}"]); sc = open_scores(X[sup], y[sup], X[qry], known)
                    for meth, (score, pred) in sc.items():
                        rec.setdefault(f"{fn}|auroc_{meth}", []).append(100 * roc_auc_score(isu, score))
                        rec.setdefault(f"{fn}|fpr95_{meth}", []).append(fpr95(isu, score))
                        kq = ~isu; rec.setdefault(f"{fn}|closed_bal_acc_{meth}", []).append(100 * balanced_accuracy_score(y[qry][kq], pred[kq]))
            out[f"fieldB|k{k}"] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in rec.items()}
            RAW[f"fieldB|k{k}"] = {m: [float(x) for x in v] for m, v in rec.items()}
    # (3) second crop (PlantDoc tomato, field C): each of the 8 classes held out in turn; tomato bank of 16 descriptions
    fp = os.path.join(OUT_CACHE, "fs_C.npz")
    if os.path.exists(fp):
        Z = np.load(fp); y, g = Z["y"], Z["g"]; tnames = TOMATO_CLASSES + list(TOMATO_EXTRA_BANK)
        for mdl in [k[5:] for k in Z.files if k.startswith("clip_")]:
            X, bank = Z[f"clip_{mdl}"], Z[f"bank_{mdl}"]; mu_img, mu_txt = X.mean(0), bank.mean(0)
            tau = taus.get(mdl, {}).get("chosen", 4)                       # tau chosen on okra, transferred unchanged
            for k in [k for k in (1, 2, 5) if k <= max_k(y, g)]:
                rec = {}
                for u in range(len(TOMATO_CLASSES)):
                    known = np.array([c for c in range(len(TOMATO_CLASSES)) if c != u]); keep = y != u
                    for d in range(3 if QUICK else 10):
                        rng = np.random.default_rng(700 + 10 * u + d)
                        sup_k, _ = draw(y[keep], g[keep], k, rng); sup = np.where(keep)[0][sup_k]
                        qry = np.where(~np.isin(g, g[sup]))[0]; isu = (y[qry] == u)
                        sc = open_scores(X[sup], y[sup], X[qry], known, bank[known], tau, mu_img, mu_txt)
                        for meth, (score, pred) in sc.items():
                            rec.setdefault(f"auroc_{meth}", []).append(100 * roc_auc_score(isu, score))
                            rec.setdefault(f"fpr95_{meth}", []).append(fpr95(isu, score))
                            rec.setdefault(f"closed_bal_acc_{meth}", []).append(100 * balanced_accuracy_score(y[qry][~isu], pred[~isu]))
                        per, grp, top3 = name_unknown(X[qry][isu], bank, [tnames[c] for c in known], mu_img, mu_txt, names=tnames)
                        nm = tnames[u]
                        rec.setdefault("name_img_top1", []).append(100 * np.mean([p == nm for p in per]))
                        rec.setdefault("name_group_top1", []).append(100.0 * (grp == nm))
                        rec.setdefault("name_group_top3", []).append(100.0 * (nm in top3))
                        rec.setdefault(f"per_class_name_group_{nm}", []).append(100.0 * (grp == nm))
                        rec.setdefault(f"per_class_name_top3_{nm}", []).append(100.0 * (nm in top3))
                        rec.setdefault(f"per_class_auroc_symprior_{nm}", []).append(100 * roc_auc_score(isu, sc["symprior"][0]))
                        idx = np.argsort(-sc["symprior"][0])[:int(round(0.2 * len(qry)))]
                        _, gf, t3f = name_unknown(X[qry][idx], bank, [tnames[c] for c in known], mu_img, mu_txt, names=tnames)
                        rec.setdefault("flag20_precision", []).append(100 * isu[idx].mean())
                        rec.setdefault("flag20_name_top1", []).append(100.0 * (gf == nm)); rec.setdefault("flag20_name_top3", []).append(100.0 * (nm in t3f))
                out[f"fieldC_loco|{mdl}|k{k}"] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in rec.items()}
                RAW[f"fieldC_loco|{mdl}|k{k}"] = {m: [float(x) for x in v] for m, v in rec.items()}
    out["_bank"] = bank_names(); save_json(out, "fs_openset.json"); save_json(RAW, "fs_openset_raw.json"); print("saved", os.path.join(OUT_RES, "fs_openset.json"))


def stage_bank():
    """Add the description-bank text embeddings to fs_source_clip.npz (text encoder only; under a minute)."""
    import fm_text
    fp = os.path.join(OUT_CACHE, "fs_source_clip.npz"); S = dict(np.load(fp))
    clips, DEV = fm_text.load_models()
    for name, (m, pre, tok) in clips.items():
        S[f"txt_{name}"] = clip_text(m, tok, DEV, [fm_text.PROMPTS[c] for c in fm_text.PROMPTS] + [YVMV_PROMPTS])
        S[f"bank_{name}"] = np.concatenate([S[f"txt_{name}"], clip_text(m, tok, DEV, list(EXTRA_BANK.values()))])
        print(name, "bank", S[f"bank_{name}"].shape)
    np.savez_compressed(fp, **S); print("bank saved to", fp)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "embed":
        stage_embed(sys.argv[2], sys.argv[3])
    elif st == "probe":
        stage_probe()
    elif st == "finetune":
        stage_finetune()
    elif st == "openset":
        stage_openset()
    elif st == "bank":
        stage_bank()
