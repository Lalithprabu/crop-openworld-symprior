"""All numbers, tables and figures of conference paper 2, computed from the saved results (no number is typed by hand).

Inputs : results/fs_probe.json (+ _raw), fs_finetune.json, fs_openset.json (+ _raw), cache/fs_*.npz
Outputs: results/paper_numbers.json, manuscript/fig_*.pdf
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
P, PR, FT, OS, OSR = R("fs_probe.json"), R("fs_probe_raw.json"), R("fs_finetune.json"), R("fs_openset.json"), R("fs_openset_raw.json")
MS = os.path.join(ROOT, "manuscript"); os.makedirs(MS, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
N = {}


def ba(ds, key):
    return [r["bal_acc"] for r in PR[ds][key]]


def ci(v, B=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed); bs = rng.choice(v, (B, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ks(ds):
    return sorted({int(k.rsplit("|", 1)[1]) for k in PR[ds]})


BASE = ["probe", "centroid", "tipadapter", "tipadapter_gap", "alignmix", "centroid_gap", "probe_gap", "zeroshot"]   # mixfix_gap is an ablation, not a baseline          # every non-SymPrior method, every feature extractor
FEATS = {"clip_bioclip2": "BioCLIP-2", "clip_bioclip": "BioCLIP", "dinov2": "DINOv2", "mnv3": "MobileNetV3 (ImageNet)",
         "source": "MobileNetV3 (Okra DiseaseNet)"}

# ---------------------------------------------------------------- 1. few-shot learning: SymPrior vs the best baseline per k
for ds in [d for d in ("A", "B", "C") if d in PR]:
    for k in ks(ds):
        sym = f"clip_bioclip2|symprior|{k}"
        cands = [key for key in PR[ds] if key.endswith(f"|{k}") and key.split("|")[1] in BASE]
        best = max(cands, key=lambda c: np.mean(ba(ds, c)))
        m, lo, hi = ci(ba(ds, sym)); bm = np.mean(ba(ds, best))
        rec = {"symprior": [m, lo, hi], "best_baseline": best, "best_baseline_mean": float(bm)}
        if k > 0:
            d = np.array(ba(ds, sym)) - np.array(ba(ds, best))
            rec.update({"diff_mean": float(d.mean()), "diff_ci": ci(d)[1:], "wins": int((d > 0).sum()), "n": int(len(d)),
                        "p_wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
            g = np.array(ba(ds, sym)) - np.array(ba(ds, f"clip_bioclip2|symprior_nogap|{k}"))
            rec["gap_gain"] = float(g.mean()); rec["gap_p"] = float(wilcoxon(g).pvalue) if np.any(g != 0) else 1.0
        if k > 0:                                                    # same encoder: SymPrior vs linear probe on BioCLIP-2
            d2 = np.array(ba(ds, sym)) - np.array(ba(ds, f"clip_bioclip2|probe|{k}"))
            rec["vs_same_probe"] = float(d2.mean()); rec["vs_same_probe_p"] = float(wilcoxon(d2).pvalue) if np.any(d2 != 0) else 1.0
            rec["vs_same_probe_wins"] = int((d2 > 0).sum())
            d3 = np.array(ba(ds, f"clip_bioclip2|symprior_inductive|{k}")) - np.array(ba(ds, best))
            rec["inductive_vs_best"] = float(d3.mean()); rec["inductive_vs_best_p"] = float(wilcoxon(d3).pvalue) if np.any(d3 != 0) else 1.0
        if k > 0:                                                    # ablation: the same gap-corrected prototypes with ONE fixed weight
            d4 = np.array(ba(ds, sym)) - np.array(ba(ds, f"clip_bioclip2|mixfix_gap|{k}"))
            rec["vs_mixfix"] = float(d4.mean()); rec["vs_mixfix_p"] = float(wilcoxon(d4).pvalue) if np.any(d4 != 0) else 1.0
        rec["inductive"] = float(np.mean(ba(ds, f"clip_bioclip2|symprior_inductive|{k}")))
        rec["zeroshot"] = float(np.mean(ba(ds, f"clip_bioclip2|zeroshot|{k}")))
        rec["nogap"] = float(np.mean(ba(ds, f"clip_bioclip2|symprior_nogap|{k}")))
        if ds == "B":
            rec["yvmv_recall_symprior"] = float(np.mean([r["yvmv_recall"] for r in PR[ds][sym]]))
        N[f"{ds}|k{k}"] = rec
N["tau"] = {m: v["chosen"] for m, v in P["_tau"].items()}
N["info"] = {ds: P[ds]["_info"] for ds in ("A", "B", "C") if ds in P}
N["finetune"] = {ds: {k: v["bal_acc"] for k, v in FT[ds].items()} for ds in FT}

# ---------------------------------------------------------------- 2. open world
N["open"] = {}
for key, v in OS.items():
    if key.startswith("_"):
        continue
    N["open"][key] = {m: x for m, x in v.items()}
for k in (1, 2):
    raw = OSR.get(f"fieldB|k{k}")
    if raw:
        for other in ("clip:bioclip2|auroc_centroid", "clip:bioclip2|auroc_msp", "clip:bioclip2|auroc_knn", "clip:bioclip2|auroc_mcm"):
            d = np.array(raw["clip:bioclip2|auroc_symprior"]) - np.array(raw[other])
            N["open"][f"fieldB|k{k}"][f"symprior_vs_{other.split('_')[-1]}_p"] = float(wilcoxon(d).pvalue)
for k in (1, 2, 5):
    raw = OSR.get(f"fieldC_loco|bioclip2|k{k}")
    if raw:
        for other in ("auroc_centroid", "auroc_msp", "auroc_knn", "auroc_mcm"):
            d = np.array(raw["auroc_symprior"]) - np.array(raw[other])
            N["open"][f"fieldC_loco|bioclip2|k{k}"][f"symprior_vs_{other.split('_')[-1]}_p"] = float(wilcoxon(d).pvalue)
for k in (1, 2, 5):
    raw = OSR.get(f"source_loco|bioclip2|k{k}")
    if raw:
        for other in ("auroc_centroid", "auroc_msp", "auroc_knn", "auroc_mcm"):
            d = np.array(raw["auroc_symprior"]) - np.array(raw[other])
            N["open"][f"source_loco|bioclip2|k{k}"][f"symprior_vs_{other.split('_')[-1]}_p"] = float(wilcoxon(d).pvalue)
json.dump(N, open(os.path.join(ROOT, "results", "paper_numbers.json"), "w"), indent=1)

# ---------------------------------------------------------------- figures
C = {"sym": "#1f5fa8", "ind": "#6a9fd8", "probe": "#c0392b", "cent": "#e08e45", "tip": "#7d7d7d", "zs": "#b0b0b0",
     "ftI": "#2e8b57", "ftS": "#8fbc8f"}


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


# Fig. 1: overview on real okra field-B data. (a) pipeline, with the modality gap shown on the real embeddings (PCA);
# (b) detect, (c) name, (d) learn: real results of each step; (e, f) text-photo similarity before/after gap correction.
Z = np.load(os.path.join(ROOT, "cache", "fs_B.npz")); S = np.load(os.path.join(ROOT, "cache", "fs_source_clip.npz"))
X, y, g, T = Z["clip_bioclip2"], Z["y"], Z["g"], Z["ctxt_bioclip2"]; BANK = S["bank_bioclip2"]; mu_t = BANK.mean(0)
BNAMES = ["Healthy", "Alternaria leaf spot", "Cercospora leaf spot", "Downy mildew", "Leaf curl", "Phyllosticta leaf spot",
          "Yellow vein mosaic", "Powdery mildew", "Enation leaf curl", "Bacterial leaf spot", "Anthracnose", "Fusarium wilt",
          "Jassid hopper burn", "Spider mite damage", "Nitrogen deficiency"]
CLS_COL = ["#4c9a2a", "#e08e45", "#8c564b", "#d4a000"]; TAU = N["tau"]["bioclip2"]; FS = 6.0


def draw1(y_, g_, k, rng):                                 # identical to fewshot.draw: same seeds give the same episodes
    sup = []
    for c in np.unique(y_):
        gs_ = np.unique(g_[y_ == c]); rng.shuffle(gs_)
        for gg in gs_[:k]:
            sup.append(rng.choice(np.where((g_ == gg) & (y_ == c))[0]))
    sup = np.array(sup); return sup, np.where(~np.isin(g_, g_[sup]))[0]


def pca2(A):
    A = A - A.mean(0); U_, s_, Vt = np.linalg.svd(A, full_matrices=False); return A @ Vt[:2].T


W, H = 7.16, 3.85
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
def box(x0, y0, w, h, col, ec="#555555", lw=0.6):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=col, ec=ec, lw=lw))
arr = dict(arrowstyle="-|>", lw=0.8, color="#333333", mutation_scale=7)
def arrow(p, q, **kw):
    ax.annotate("", xy=q, xytext=p, arrowprops={**arr, **kw})
ax.text(0.03, H - 0.02, "(a) The SymPrior pipeline in a new field (okra field B, real photographs and embeddings)", fontsize=7.4, fontweight="bold", va="top")
y0b, y1b = 2.02, 3.58                                       # top band of boxes

# 1 photos
box(0.03, y0b, 1.5, y1b - y0b, "#f4f4f4")
ax.text(0.78, y1b - 0.05, "Field photos (unlabelled)", ha="center", va="top", fontsize=FS, fontweight="bold")
Xn = unit(X); thumbs = []
for c in range(4):
    idx = np.where(y == c)[0]; thumbs.append(idx[np.argmax(Xn[idx] @ unit(Xn[idx].mean(0)))])
N["fig1_thumb_index"] = [int(i) for i in thumbs]
tl = ["Healthy", "Cercospora LS", "Leaf curl", "YVMV (unseen)"]
for n_, i in enumerate(thumbs):
    r, c = divmod(n_, 2); x0, yy0 = 0.1 + c * 0.7, y1b - 0.78 - r * 0.66
    ia = fig.add_axes([x0 / W, yy0 / H, 0.64 / W, 0.5 / H]); ia.imshow(Z["I352"][i][30:322, :, ::-1], aspect="auto", interpolation="lanczos"); ia.axis("off")
    ax.text(x0 + 0.32, yy0 - 0.02, tl[n_], ha="center", va="top", fontsize=FS - 0.5, color="#b22222" if n_ == 3 else "black",
            fontweight="bold" if n_ == 3 else "normal")
# 2 descriptions
box(1.65, y0b, 1.45, y1b - y0b, "#f4f4f4")
ax.text(2.375, y1b - 0.05, "Written symptom descriptions", ha="center", va="top", fontsize=FS, fontweight="bold")
for j, (cl, d, col) in enumerate((("Healthy", "uniformly green, no spots", CLS_COL[0]), ("Cercospora LS", "sooty black angular spots", CLS_COL[1]),
                                  ("Leaf curl", "curled, crinkled, thick veins", CLS_COL[2]), ("YVMV", "bright yellow network of veins", CLS_COL[3]))):
    yy = y1b - 0.3 - j * 0.27
    ax.text(1.72, yy, f"{cl}:", fontsize=FS - 0.4, fontweight="bold", va="top", color=col if j else "#2e6b16")
    ax.text(1.72, yy - 0.1, f"“{d}”", fontsize=FS - 0.6, style="italic", va="top")
ax.text(2.375, y0b + 0.08, "+ 8 look-alikes → bank of 15 names", ha="center", fontsize=FS - 0.8, color="#555555")
# 3 encoder
box(3.22, 2.32, 0.62, 0.95, "#dbe7f5")
ax.text(3.53, 2.795, "BioCLIP-2\nimage +\ntext\nencoders", ha="center", va="center", fontsize=FS - 0.3)
ax.plot([0.78, 0.78, 3.53], [y1b + 0.02, 3.635, 3.635], color="#333333", lw=0.8); arrow((3.53, 3.635), (3.53, 3.27)); arrow((3.10, 2.795), (3.22, 2.795))
# 4 modality gap with real PCA
box(3.96, y0b, 1.62, y1b - y0b, "#dbe7f5")
ax.text(4.77, y1b - 0.05, "Modality-gap correction (no labels)", ha="center", va="top", fontsize=FS, fontweight="bold")
for j, (ttl, Xp, Bp) in enumerate((("raw", unit(X), unit(BANK)), ("centred", unit(X - X.mean(0)), unit(BANK - mu_t)))):
    P2 = pca2(np.vstack([Xp, Bp])); pi, pt = P2[:len(Xp)], P2[len(Xp):]
    a2 = fig.add_axes([(4.04 + j * 0.78) / W, 2.42 / H, 0.7 / W, 0.88 / H])
    for c in range(4):
        a2.scatter(pi[y == c, 0], pi[y == c, 1], s=1.2, color=CLS_COL[c], alpha=0.7, lw=0)
    a2.scatter(pt[:, 0], pt[:, 1], s=9, marker="*", color="black", lw=0)
    a2.set_xticks([]); a2.set_yticks([]); a2.set_title(ttl, fontsize=FS - 1.0, pad=1.5)
    for sp in a2.spines.values():
        sp.set_linewidth(0.4)
ax.text(4.77, 2.3, "$\\bullet$ photos (colour = class)    $\\bigstar$ descriptions", ha="center", va="top", fontsize=FS - 1.2, color="#333333")
ax.text(4.77, 2.12, "$\\tilde{\\mathbf{x}}=\\frac{\\mathbf{x}-\\boldsymbol{\\mu}_v}{\\Vert\\cdot\\Vert}$,  $\\tilde{\\mathbf{t}}=\\frac{\\mathbf{t}-\\boldsymbol{\\mu}_t}{\\Vert\\cdot\\Vert}$",
        ha="center", va="center", fontsize=FS + 0.4)
arrow((3.84, 2.795), (3.96, 2.795))
# 5 SymPrior
box(5.7, y0b, 1.43, y1b - y0b, "#e2f0d9")
ax.text(6.415, y1b - 0.05, "SymPrior (training-free)", ha="center", va="top", fontsize=FS, fontweight="bold")
ax.text(6.415, y1b - 0.42, "$\\mathbf{p}_c\\propto\\tau\\,\\tilde{\\mathbf{t}}_c+\\sum_{i\\in S_c}\\tilde{\\mathbf{x}}_i$", ha="center", va="center", fontsize=FS + 1.6)
for j, t_ in enumerate(("description = prior worth τ photos", f"τ = {TAU}, fixed on the source data", "each labelled photo adds evidence",
                        "text weight τ/(τ+k) falls with k", "no gradient step, one vector sum")):
    ax.text(5.78, y1b - 0.72 - j * 0.15, "• " + t_, fontsize=FS - 0.7, va="top")
ax.text(6.415, y0b + 0.08, "posterior prototypes → (b) (c) (d)", ha="center", fontsize=FS - 0.8, color="#1f3f73")
arrow((5.58, 2.795), (5.7, 2.795))

# bottom band: real results of each step
ax.plot([0.03, W - 0.03], [1.9, 1.9], color="#999999", lw=0.5)
ax.text(0.03, 1.87, "Results of each step on okra field B (real data; YVMV is the disease unseen in the source data)", fontsize=6.6, fontweight="bold", va="top")
# (b) detect: unknown-ness score in one k=2 episode
rng = np.random.default_rng(900 + 10 * 2 + 0); keep = np.where(y != 3)[0]; known = np.array([0, 1, 2])
sk, _ = draw1(y[keep], g[keep], 2, rng); sup = keep[sk]; qry = np.where(~np.isin(g, g[sup]))[0]
Xc, Tc = unit(X - X.mean(0)), unit(T - mu_t)
Pp = np.stack([unit(TAU * Tc[c] + Xc[sup][y[sup] == c].sum(0)) for c in known]); u = 1 - (Xc[qry] @ Pp.T).max(1); isu = y[qry] == 3
ab = fig.add_axes([0.32 / W, 0.32 / H, 1.15 / W, 1.18 / H])
bins = np.linspace(u.min(), u.max(), 18)
ab.hist(u[~isu], bins=bins, color="#7f9fbf", alpha=0.85, label="known diseases", density=True)
ab.hist(u[isu], bins=bins, color="#d4a000", alpha=0.75, label="YVMV (unknown)", density=True)
ab.set_yticks([]); ab.set_xlabel("unknown score $u=1-\\max_c\\mathbf{p}_c^{\\top}\\tilde{\\mathbf{x}}$", fontsize=FS - 0.8, labelpad=1)
ab.tick_params(labelsize=FS - 1.2, length=1.5, pad=1); ab.legend(fontsize=FS - 1.5, frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.2)
ab.set_title("(b) DETECT: one episode, $k$=2", fontsize=FS, pad=2)
ab.set_ylim(0, ab.get_ylim()[1] * 1.45)
N["fig1_detect_episode_auroc"] = float(np.mean([(a > b) + 0.5 * (a == b) for a in u[isu] for b in u[~isu]]) * 100)
# (c) name: pooled YVMV photographs, k=1 episode 0
rng = np.random.default_rng(900 + 10 * 1 + 0); sk, _ = draw1(y[keep], g[keep], 1, rng); sup = keep[sk]; qry = np.where(~np.isin(g, g[sup]))[0]
U_ = X[qry][y[qry] == 3]; sim = unit(unit(U_ - X.mean(0)).mean(0)) @ unit(BANK - mu_t).T
allowed = [i for i, n in enumerate(BNAMES) if n not in ("Healthy", "Cercospora leaf spot", "Leaf curl")]
order = sorted(allowed, key=lambda i: -sim[i])[:5]
an = fig.add_axes([2.75 / W, 0.32 / H, 0.95 / W, 1.18 / H])
an.barh(range(5), [sim[i] for i in order], color=["#d4a000" if BNAMES[i] == "Yellow vein mosaic" else "#b8b8b8" for i in order], height=0.7)
an.set_yticks(range(5)); an.set_yticklabels([BNAMES[i] for i in order], fontsize=FS - 1.0); an.invert_yaxis()
an.tick_params(labelsize=FS - 1.2, length=1.5, pad=1); an.set_xlabel("similarity of pooled photos", fontsize=FS - 0.8, labelpad=1)
an.set_title(f"(c) NAME: top 3 in {OS['fieldB|k1']['clip:bioclip2|name_group_top3'][0]:.0f}% of episodes", fontsize=FS, pad=2)
N["fig1_name_episode_top"] = [BNAMES[i] for i in order]
# (d) learn
al = fig.add_axes([4.2 / W, 0.32 / H, 1.0 / W, 1.18 / H]); kk = [0, 1, 2]
al.plot(kk, [N[f"B|k{k}"]["symprior"][0] for k in kk], "-o", color=C["sym"], ms=2.5, lw=1.0, label="SymPrior")
al.plot(kk[1:], [N[f"B|k{k}"]["best_baseline_mean"] for k in kk[1:]], "-s", color=C["probe"], ms=2.2, lw=0.9, label="best other method")
al.plot(kk, [N[f"B|k{k}"]["zeroshot"] for k in kk], ":x", color="#888888", ms=2.5, lw=0.9, label="raw zero-shot")
al.axhline(25, color="black", lw=0.4, ls=":"); al.set_xticks(kk); al.set_ylim(15, 92)
al.tick_params(labelsize=FS - 1.2, length=1.5, pad=1); al.set_xlabel("labelled photos per class $k$", fontsize=FS - 0.8, labelpad=1)
al.set_ylabel("bal. acc. (%)", fontsize=FS - 0.8, labelpad=1); al.legend(fontsize=FS - 1.5, frameon=False, loc="center right", handlelength=1.2)
al.set_title(f"(d) LEARN: {N['B|k1']['symprior'][0]:.1f}% with $k$=1", fontsize=FS, pad=2)
# (e, f) heatmaps: mean text-photo similarity per class
names = ["H", "CLS", "LC", "YVMV"]
for j, (title, Xp, Tp) in enumerate((("(e) raw", unit(X), unit(T)), ("(f) gap-corrected", unit(X - X.mean(0)), unit(T - mu_t)))):
    M = np.stack([(Xp[y == c] @ Tp.T).mean(0) for c in range(4)])
    acc = 100 * np.mean([np.mean((Xp[y == c] @ Tp.T).argmax(1) == c) for c in range(4)])
    a2 = fig.add_axes([(5.72 + j * 0.72) / W, 0.32 / H, 0.62 / W, 1.18 / H])
    a2.imshow(M, cmap="Blues", vmin=M.min(), vmax=M.max(), aspect="auto")
    for i in range(4):
        for jj in range(4):
            a2.text(jj, i, f"{M[i, jj]:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=FS - 1.9,
                    fontweight="bold" if i == jj else "normal", color="white" if M[i, jj] > M.min() + 0.6 * (M.max() - M.min()) else "black")
    a2.set_xticks(range(4)); a2.set_xticklabels(names, fontsize=FS - 1.4, rotation=90); a2.set_yticks(range(4))
    a2.set_yticklabels(names if j == 0 else [], fontsize=FS - 1.4); a2.tick_params(length=1.5, pad=1)
    a2.set_xlabel("description", fontsize=FS - 1.0, labelpad=1)
    if j == 0:
        a2.set_ylabel("photos", fontsize=FS - 1.0, labelpad=1)
    a2.set_title(f"{title}\nzero-shot {acc:.1f}%", fontsize=FS - 0.6, pad=2)
    N[f"fig1_{'raw' if j == 0 else 'corrected'}_zeroshot_balacc"] = float(acc)
fig.savefig(os.path.join(MS, "fig_concept.pdf"), bbox_inches=None); fig.savefig(os.path.join(MS, "fig_concept.png"), dpi=300, bbox_inches=None); plt.close(fig)
json.dump(N, open(os.path.join(ROOT, "results", "paper_numbers.json"), "w"), indent=1)

# Fig. 2: learning curves with 95% CIs
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.1), gridspec_kw={"width_ratios": [1.25, 0.8, 1.15]})
for ax, ds, ttl in zip(axs, ("A", "B", "C"), ("Okra field A (2 classes)", "Okra field B (4 classes)", "Tomato, PlantDoc (8 classes)")):
    kk = ks(ds)
    def curve(key, lab, col, ls="-", mk="o"):
        pts = [(k, *ci(ba(ds, f"{key}|{k}"))) for k in kk if f"{key}|{k}" in PR[ds]]
        if not pts:
            return
        xs = [p[0] for p in pts]; xp = [max(x, 0.0) for x in xs]
        xpos = [x if x > 0 else 0.5 for x in xp]
        ax.plot(xpos, [p[1] for p in pts], ls=ls, marker=mk, ms=2.5, lw=1.0, color=col, label=lab)
        ax.fill_between(xpos, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.15, lw=0)
    curve("clip_bioclip2|symprior", "SymPrior (ours)", C["sym"])
    curve("clip_bioclip2|symprior_inductive", "SymPrior, inductive (ours)", C["ind"], "--")
    bestprobe = max([f for f in FEATS if f"{f}|probe|1" in PR[ds]], key=lambda f: np.mean(ba(ds, f"{f}|probe|1")))
    curve(f"{bestprobe}|probe", "Linear probe (best encoder)", C["probe"]); N[f"fig2_bestprobe_{ds}"] = FEATS[bestprobe]
    curve("clip_bioclip2|centroid", "Prototype (BioCLIP-2)", C["cent"], ":")
    curve("clip_bioclip2|tipadapter", "Tip-Adapter (BioCLIP-2)", C["tip"], "-.")
    curve("clip_bioclip2|tipadapter_gap", "Tip-Adapter, gap-corr. (BioCLIP-2)", "#9b59b6", "-.", "^")
    curve("clip_bioclip2|alignmix", "Align+Mix (BioCLIP-2)", "#17becf", "--", "v")
    curve("clip_bioclip2|zeroshot", "Zero-shot text (BioCLIP-2)", C["zs"], ":", "x")
    for init, col, lab in ((("imagenet", C["ftI"], "Fine-tune (ImageNet init)"), ("source", C["ftS"], "Fine-tune (source init)")) if ds in FT else ()):
        fk = sorted(int(k.rsplit("|", 1)[1]) for k in FT[ds] if f"_{init}|" in k)
        ax.errorbar(fk, [FT[ds][f"mnv3_ft_{init}|ft|{k}"]["bal_acc"][0] for k in fk],
                    yerr=[FT[ds][f"mnv3_ft_{init}|ft|{k}"]["bal_acc"][1] for k in fk], color=col, lw=0.9, marker="s", ms=2.5,
                    capsize=1.5, label=lab)
    ax.set_xscale("log"); xt = [0.5] + [k for k in kk if k > 0]; ax.set_xticks(xt); ax.set_xticklabels(["0"] + [str(k) for k in kk if k > 0])
    ax.minorticks_off(); ax.set_xlabel("Labelled photographs per class ($k$)"); ax.set_ylabel("Balanced accuracy (%)")
    ax.axhline({"A": 50, "B": 25, "C": 12.5}[ds], color="black", lw=0.5, ls=":"); ax.set_title(ttl, fontsize=7)
    if ds != "A":
        ax.set_ylabel("")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.8, rect=(0, 0.19, 1, 1))
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=6.0, columnspacing=1.2, handlelength=2.2); fig.savefig(os.path.join(MS, "fig_curves.pdf")); plt.close(fig)
json.dump(N, open(os.path.join(ROOT, "results", "paper_numbers.json"), "w"), indent=1)

# Fig. 3: open world
fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.95), gridspec_kw={"width_ratios": [1.5, 1.1, 1.2]})
meths = [("symprior", "SymPrior (ours)", C["sym"]), ("centroid", "Prototype", C["cent"]), ("knn", "Nearest neighbour", C["tip"]), ("msp", "Max. softmax prob.", C["probe"])]
groups = [("source_loco|bioclip2|k5", "", "DiseaseNet\n$k$=5"), ("fieldB|k1", "clip:bioclip2|", "Field B\n$k$=1"),
          ("fieldB|k2", "clip:bioclip2|", "Field B\n$k$=2"), ("fieldC_loco|bioclip2|k5", "", "Tomato\n$k$=5")]
w = 0.19
for gi, (gk, pre, lab) in enumerate(groups):
    for mi, (m, ml, col) in enumerate(meths):
        v = OS[gk][f"{pre}auroc_{m}"]
        axs[0].bar(gi + (mi - 1.5) * w, v[0], w, yerr=v[1], color=col, capsize=1, error_kw={"lw": 0.5}, label=ml if gi == 0 else None)
axs[0].set_xticks(range(len(groups))); axs[0].set_xticklabels([g[2] for g in groups], fontsize=6)
axs[0].set_ylim(40, 112); axs[0].set_yticks([40, 50, 60, 70, 80, 90, 100]); axs[0].axhline(50, color="black", lw=0.5, ls=":"); axs[0].set_ylabel("AUROC (%)")
axs[0].set_title("(a) Detect: unknown vs known"); axs[0].legend(frameon=False, fontsize=5.4, ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.8)
cats = [("Per photo", "name_img_top1"), ("Pooled, top-1", "name_group_top1"), ("Pooled, top-3", "name_group_top3"),
        ("Flagged 20%, top-1", "flag20_name_top1"), ("Flagged 20%, top-3", "flag20_name_top3")]
hb = 0.38
for kj, (kk_, col) in enumerate(((1, "#9dbbe0"), (2, C["sym"]))):
    vals = [OS[f"fieldB|k{kk_}"][f"clip:bioclip2|{key}"][0] for _, key in cats]
    axs[1].barh(np.arange(len(cats)) + (kj - 0.5) * hb, vals, hb, color=col, label=f"$k$={kk_}")
    for i, v in enumerate(vals):
        axs[1].text(v + 2, i + (kj - 0.5) * hb, f"{v:.0f}", va="center", fontsize=4.8)
axs[1].set_yticks(range(len(cats))); axs[1].set_yticklabels([c for c, _ in cats], fontsize=5.6); axs[1].invert_yaxis(); axs[1].set_xlim(0, 118)
axs[1].axvline(100 / 12, color="black", lw=0.5, ls=":"); axs[1].axvline(25, color="black", lw=0.5, ls="--")
axs[1].set_xlabel("Named correctly (%)"); axs[1].set_title("(b) Name unseen YVMV (field B)")
axs[1].legend(frameon=False, fontsize=5.2, loc="upper right", handlelength=1.0)
src = OS["source_loco|bioclip2|k5"]; tom = OS["fieldC_loco|bioclip2|k5"]
okra_c = [("Healthy", "Healthy"), ("Alternaria leaf spot", "Alternaria LS"), ("Cercospora leaf spot", "Cercospora LS"), ("Downy mildew", "Downy mildew"),
          ("Leaf curl", "Leaf curl"), ("Phyllosticta leaf spot", "Phyllosticta LS")]
tom_c = [("Healthy", "Healthy"), ("Early blight", "Early blight"), ("Septoria leaf spot", "Septoria LS"), ("Bacterial spot", "Bacterial spot"),
         ("Late blight", "Late blight"), ("Mosaic virus", "Mosaic virus"), ("Yellow leaf curl virus", "Yellow leaf curl"), ("Leaf mold", "Leaf mold")]
labs = [f"Okra: {s_}" if i == 0 else s_ for i, (_, s_) in enumerate(okra_c)] + [f"Tomato: {s_}" if i == 0 else s_ for i, (_, s_) in enumerate(tom_c)]
v1 = [src[f"per_class_name_group_{c}"][0] for c, _ in okra_c] + [tom[f"per_class_name_group_{c}"][0] for c, _ in tom_c]
cols = [C["sym"]] * len(okra_c) + ["#c0392b"] * len(tom_c); yy = list(range(len(okra_c))) + [len(okra_c) + 0.6 + i for i in range(len(tom_c))]
axs[2].barh(yy, v1, color=cols, height=0.75); axs[2].set_yticks(yy); axs[2].set_yticklabels(labs, fontsize=5.0); axs[2].invert_yaxis()
axs[2].set_xlim(0, 118); axs[2].set_xlabel("Pooled top-1 naming (%)"); axs[2].set_title("(c) Name held-out class, $k$=5")
for i, v in zip(yy, v1):
    axs[2].text(v + 2, i, f"{v:.0f}", va="center", fontsize=4.8)
fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(MS, "fig_openworld.pdf")); plt.close(fig)
print(json.dumps({k: v for k, v in N.items() if k.startswith(("A|", "B|"))}, indent=0)[:3000])
