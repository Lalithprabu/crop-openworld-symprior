"""SymPrior-T and one-click-per-class labelling (PREREGISTRATION.md, sections 2 and 3).
Hyperparameters chosen on the source episodes only, then fixed for fields A, B, C.
Usage: python symprior_t.py        -> results/fs_symprior_t.json
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.cluster import KMeans
import fewshot as F

unit, draw = F.unit, F.draw
TAU, ITERS = 4.0, 10
KAPPAS, GAMMAS = [10, 20, 50, 100], [0.5, 1, 2, 4, 8, 16]


def posterior(Xs, ys, Xu, T, cls, kappa, gamma, tau=TAU, iters=ITERS):
    """Xs, Xu, T already gap-corrected and unit. Returns class directions P (C x d)."""
    base = np.stack([tau * T[j] + Xs[ys == c].sum(0) for j, c in enumerate(cls)])
    P = unit(base)
    if gamma > 0:
        C, Nu = len(cls), len(Xu)
        for _ in range(iters):
            L = kappa * Xu @ P.T; L -= L.max(1, keepdims=True); R = np.exp(L); R /= R.sum(1, keepdims=True)
            P = unit(base + gamma * (R.T @ Xu) / (Nu / C))
    return P


def probs(X, P, kappa):
    L = kappa * X @ P.T; L -= L.max(1, keepdims=True); R = np.exp(L); return R / R.sum(1, keepdims=True)


def gapc(X, T, mu_img, mu_txt):
    return unit(X - mu_img), unit(T - mu_txt)


def source_select():
    S = np.load(os.path.join(F.OUT_CACHE, "fs_source_clip.npz"))
    X, y, g = S["img_bioclip2"], S["y"], S["g"]; cls = np.arange(6)
    Xg, Tg = gapc(X, S["txt_bioclip2"][:6], X.mean(0), S["bank_bioclip2"].mean(0))
    score = {(ka, ga): {} for ka in KAPPAS for ga in [0] + GAMMAS}
    for k in (0, 1, 2, 5, 10):
        for d in range(1 if k == 0 else 20):
            sup, qry = draw(y, g, k, np.random.default_rng(77 + 100 * k + d)) if k else (np.array([], int), np.arange(len(y)))
            for (ka, ga) in score:
                P = posterior(Xg[sup], y[sup], Xg[qry], Tg, cls, ka, ga)
                score[(ka, ga)].setdefault(k, []).append(bacc(y[qry], cls[(Xg[qry] @ P.T).argmax(1)]))
    mean = {key: float(100 * np.mean([np.mean(v) for v in s.values()])) for key, s in score.items()}
    best = max([kk for kk in mean if kk[1] > 0], key=mean.get)
    return best, {f"kappa{a}|gamma{b}": v for (a, b), v in mean.items()}, \
        {f"k{k}": float(100 * np.mean(score[best][k])) for k in score[best]}, {f"k{k}": float(100 * np.mean(score[(10, 0)][k])) for k in score[best]}


def field(ds):
    Z = np.load(os.path.join(F.OUT_CACHE, f"fs_{ds}.npz")); S = np.load(os.path.join(F.OUT_CACHE, "fs_source_clip.npz"))
    y, g = Z["y"], Z["g"]; cls = np.unique(y); X = Z["clip_bioclip2"]
    mu_txt = Z["bank_bioclip2"].mean(0) if "bank_bioclip2" in Z.files else S["bank_bioclip2"].mean(0)
    Xg, Tg = gapc(X, Z["ctxt_bioclip2"][cls], X.mean(0), mu_txt)
    return y, g, cls, Xg, Tg


def evaluate(kappa, gamma):
    raw = json.load(open(os.path.join(F.OUT_RES, "fs_probe_raw.json")))
    out = {}
    for ds in ("A", "B", "C"):
        y, g, cls, Xg, Tg = field(ds); R = raw[ds]; o = {}
        for k in [k for k in (0, 1, 2, 5, 10, 20) if k <= F.max_k(y, g)]:
            st, ss = [], []
            for d in range(1 if k == 0 else F.N_DRAWS):
                sup, qry = draw(y, g, k, np.random.default_rng(1000 * k + d)) if k else (np.array([], int), np.arange(len(y)))
                P = posterior(Xg[sup], y[sup], Xg[qry], Tg, cls, kappa, gamma)
                st.append(100 * bacc(y[qry], cls[(Xg[qry] @ P.T).argmax(1)]))
                P0 = posterior(Xg[sup], y[sup], Xg[qry], Tg, cls, kappa, 0)
                ss.append(100 * bacc(y[qry], cls[(Xg[qry] @ P0.T).argmax(1)]))
            sp = [r["bal_acc"] for r in R[f"clip_bioclip2|symprior|{k}"]]
            assert np.allclose(sp, ss, atol=1e-6), (ds, k, sp[:3], ss[:3])          # our SymPrior re-implementation = paper numbers
            base = {}
            for key, v in R.items():
                f_, m, kk = key.split("|")
                if int(kk) == k and "tuned" not in m and "symprior" not in m and m != "mixfix_gap":
                    base[f"{f_}|{m}"] = [r["bal_acc"] for r in v]
            best = max(base, key=lambda b: np.mean(base[b]))
            e = {"symprior_t": [float(np.mean(st)), float(np.std(st))], "symprior": [float(np.mean(ss)), float(np.std(ss))],
                 "best_baseline": best, "best_baseline_acc": float(np.mean(base[best]))}
            if k > 0:
                e["p_vs_symprior"] = float(wilcoxon(st, ss).pvalue) if not np.allclose(st, ss) else 1.0
                e["p_vs_best"] = float(wilcoxon(st, base[best]).pvalue)
                e["wins_vs_best"] = int(np.sum(np.array(st) > np.array(base[best])))
            o[f"k{k}"] = e
        out[ds] = o
    return out


def clicks(kappa, gamma):
    """One (two) clicks per class: description-guided choice vs k-means medoids vs random (the paper's episodes)."""
    out = {}
    for ds in ("A", "B", "C"):
        y, g, cls, Xg, Tg = field(ds); C = len(cls); o = {}
        P = posterior(Xg[:0], y[:0], Xg, Tg, cls, kappa, gamma); R = probs(Xg, P, kappa)
        for budget in (1, 2):
            if budget > F.max_k(y, g): continue
            pick, used = [], set()
            for j in range(C):                                    # the most confident photographs of each class, distinct groups
                got = 0
                for i in np.argsort(-R[:, j]):
                    if int(i) in pick or g[i] in used: continue
                    pick.append(int(i)); used.add(g[i]); got += 1
                    if got == budget: break
            km = KMeans(C * budget, n_init=1, random_state=0).fit(Xg)
            med = []
            for c in range(C * budget):
                idx = np.where(km.labels_ == c)[0]; i = idx[np.argmax(Xg[idx] @ unit(km.cluster_centers_[c]))]
                if g[i] not in g[med]: med.append(int(i))
            res = {}
            for name, sup in (("description_guided", np.array(pick)), ("kmeans_medoids", np.array(med))):
                qry = np.where(~np.isin(g, g[sup]))[0]
                Pp = posterior(Xg[sup], y[sup], Xg[qry], Tg, cls, kappa, gamma)
                res[name] = {"bal_acc": float(100 * bacc(y[qry], cls[(Xg[qry] @ Pp.T).argmax(1)])),
                             "classes_hit": int(len(np.unique(y[sup]))), "n_clicks": int(len(sup))}
            rnd = []
            for d in range(F.N_DRAWS):
                sup, qry = draw(y, g, budget, np.random.default_rng(1000 * budget + d))
                Pp = posterior(Xg[sup], y[sup], Xg[qry], Tg, cls, kappa, gamma); rnd.append(100 * bacc(y[qry], cls[(Xg[qry] @ Pp.T).argmax(1)]))
            res["random_balanced"] = {"mean": float(np.mean(rnd)), "sd": float(np.std(rnd)), "max": float(np.max(rnd))}
            res["description_guided"]["percentile_vs_random"] = float(100 * np.mean(np.array(rnd) < res["description_guided"]["bal_acc"]))
            o[f"budget{budget}"] = res
        out[ds] = o
    return out


if __name__ == "__main__":
    (ka, ga), grid, src_best, src_sym = source_select()
    print("chosen on source: kappa", ka, "gamma", ga); print("source per k  SymPrior-T", src_best, "\n              SymPrior  ", src_sym)
    res = {"_chosen": {"kappa": ka, "gamma": ga}, "_source_grid": grid, "_source_per_k": {"symprior_t": src_best, "symprior": src_sym}}
    res["fewshot"] = evaluate(ka, ga); res["clicks"] = clicks(ka, ga)
    for ds, o in res["fewshot"].items():
        for k, e in o.items():
            print(ds, k, f"T {e['symprior_t'][0]:.1f}  S {e['symprior'][0]:.1f}  best {e['best_baseline']} {e['best_baseline_acc']:.1f}",
                  {kk: round(v, 4) for kk, v in e.items() if kk.startswith("p_")}, e.get("wins_vs_best", ""))
    print(json.dumps(res["clicks"], indent=1))
    F.save_json(res, "fs_symprior_t.json")
