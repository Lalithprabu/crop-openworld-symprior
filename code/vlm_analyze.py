"""Compares the VLM answers (results_vlm/*.jsonl) with SymPrior / SymPrior-T on exactly the same photographs and episodes,
and runs the pre-registered hybrid (section 4) and distillation (section 5). Works on partial runs (only answered photographs).
Usage: python vlm_analyze.py [folder with the jsonl files]   -> results/fs_vlm.json
"""
import os, sys, glob, json
import numpy as np
from sklearn.metrics import balanced_accuracy_score as bacc
import fewshot as F, symprior_t as ST, vlm_eval as V

KAPPA, GAMMA = 50, 0.5                                    # chosen on source (results/fs_symprior_t.json)
VDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(F.ROOT, "results_vlm")


def row_of(key_part):
    return int(key_part.split(".")[0])


def load_vlm():
    out = {}
    for fn in sorted(glob.glob(os.path.join(VDIR, "*.jsonl"))):
        D = {}
        for line in open(fn, encoding="utf-8"):
            d = json.loads(line)
            if not d["raw"].startswith("ERROR"): D[d["key"]] = d
        out[os.path.basename(fn)[:-6]] = D
    return out


def bank_embed(ds):
    S = np.load(os.path.join(F.OUT_CACHE, "fs_source_clip.npz")); Z = np.load(os.path.join(F.OUT_CACHE, f"fs_{ds}.npz"))
    B = Z["bank_bioclip2"] if "bank_bioclip2" in Z.files else S["bank_bioclip2"]
    return ST.unit(B - B.mean(0)), list(V.BANK[ds])


def mcnemar(a, b):
    """exact two-sided McNemar test on paired correctness vectors a, b"""
    from scipy.stats import binomtest
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def ba(y, p):
    return float(100 * bacc(y, p)) if len(y) else None


def main():
    VL = load_vlm(); res = {"_n_answers": {m: len(D) for m, D in VL.items()}}
    for ds in ("B", "C", "A"):
        y, g, cls, Xg, Tg = ST.field(ds); rows, names = V.load_index(ds); name2c = {n: i for i, n in enumerate(names)}
        P0 = ST.posterior(Xg[:0], y[:0], Xg, Tg, cls, KAPPA, 0); PT = ST.posterior(Xg[:0], y[:0], Xg, Tg, cls, KAPPA, GAMMA)
        ours0, oursT = cls[(Xg @ P0.T).argmax(1)], cls[(Xg @ PT.T).argmax(1)]; probT = ST.probs(Xg, PT, KAPPA).max(1)
        R = {}
        for m, D in VL.items():
            r = {}
            for task in ("T1", "T2"):
                idx = [row_of(k.split("|")[2]) for k in D if k.startswith(f"{task}|{ds}|")]
                if not idx: continue
                idx = np.array(sorted(idx)); key = lambda i: f"{task}|{ds}|{rows[i].get('idx', rows[i].get('img'))}"
                pv = np.array([name2c.get(D[key(i)]["pred"], -1) for i in idx])
                r[task] = {"n": int(len(idx)), "vlm": ba(y[idx], pv), "symprior_k0": ba(y[idx], ours0[idx]), "symprior_t_k0": ba(y[idx], oursT[idx]),
                           "p_mcnemar_vlm_vs_symprior_t": mcnemar(pv == y[idx], oursT[idx] == y[idx]),
                           "n_invalid": int(np.sum(pv < 0)),
                           **({} if (pv >= 0).all() else {
                               "vlm_valid": ba(y[idx][pv >= 0], pv[pv >= 0]), "symprior_k0_valid": ba(y[idx][pv >= 0], ours0[idx][pv >= 0]),
                               "symprior_t_k0_valid": ba(y[idx][pv >= 0], oursT[idx][pv >= 0]),
                               "p_mcnemar_valid": mcnemar((pv == y[idx])[pv >= 0], (ours0[idx] == y[idx])[pv >= 0])}),
                           "p_mcnemar_vlm_vs_symprior": mcnemar(pv == y[idx], ours0[idx] == y[idx]),
                           "confusion_vlm": np.array([[int(np.sum((y[idx] == a) & (pv == b))) for b in range(len(cls))] for a in range(len(cls))]).tolist()}
                if task == "T2":                          # hybrid (section 4) and distillation (section 5)
                    hyb = np.where(probT[idx] < 0.5, pv, oursT[idx])
                    r[task]["hybrid"] = ba(y[idx], hyb); r[task]["deferred_frac"] = float(np.mean(probT[idx] < 0.5))
                    curve = {}
                    for q in (0, 10, 20, 30, 50, 70, 100):
                        thr = np.percentile(probT[idx], q) if q < 100 else np.inf
                        curve[q] = ba(y[idx], np.where(probT[idx] < thr, pv, oursT[idx]))
                    r[task]["defer_curve_pct"] = curve
                    ok = pv >= 0
                    for var in ("all", "agree"):
                        sel = ok.copy()
                        if var == "agree":
                            k1 = {row_of(k.split("|")[2]): name2c.get(d["pred"], -1) for k, d in D.items() if k.startswith(f"T1|{ds}|")}
                            sel &= np.array([k1.get(i, -2) == p for i, p in zip(idx, pv)])
                        li, lp = idx[sel], pv[sel]
                        Pd = ST.unit(np.stack([4 * Tg[j] + Xg[li][lp == j].sum(0) for j in range(len(cls))]))
                        pdl = cls[(Xg[idx] @ Pd.T).argmax(1)]
                        r[task][f"distil_{var}_transductive"] = ba(y[idx], pdl)
                        r[task][f"distil_{var}_p_vs_symprior"] = mcnemar(pdl == y[idx], ours0[idx] == y[idx])
                        r[task][f"distil_{var}_p_vs_vlm"] = mcnemar(pdl == y[idx], pv == y[idx])
                        r[task][f"distil_{var}_n_pseudo"] = int(sel.sum()); r[task][f"distil_{var}_pseudo_acc"] = float(100 * np.mean(y[li] == lp)) if len(li) else None
                        if len(idx) == len(y):            # inductive: pseudo-labels on half of the groups, test on the other half
                            accs, accv = [], []
                            for s in range(10):
                                ug = np.unique(g); rng = np.random.default_rng(s); half = set(rng.permutation(ug)[: len(ug) // 2])
                                tr = np.array([g[i] in half for i in idx]); te = ~tr
                                Pd = ST.unit(np.stack([4 * Tg[j] + Xg[idx[tr & sel]][pv[tr & sel] == j].sum(0) for j in range(len(cls))]))
                                accs.append(ba(y[idx[te]], cls[(Xg[idx[te]] @ Pd.T).argmax(1)])); accv.append(ba(y[idx[te]], pv[te]))
                            r[task][f"distil_{var}_inductive"] = [float(np.mean(accs)), float(np.std(accs))]
                            r[task]["vlm_on_inductive_halves"] = float(np.mean(accv))
            # T3 naming: top-1 / top-3 per photograph vs our per-photograph naming on the same option set
            ks = [k for k in D if k.startswith(f"T3|{ds}|")]
            if ks:
                Bk, bn = bank_embed(ds); t1v = t3v = t1o = t3o = 0; nv = v3 = o3v = 0
                for k in ks:
                    d = D[k]; i = row_of(k.split("|")[2]); opts = d["options"]
                    a = d["ans"] or {}; top = [a.get("first"), a.get("second"), a.get("third")]
                    t1v += top[0] == d["truth"]; t3v += d["truth"] in top
                    sims = {o: float(Xg[i] @ Bk[bn.index(o)]) for o in opts}; rk = sorted(opts, key=lambda o: -sims[o])
                    t1o += rk[0] == d["truth"]; t3o += d["truth"] in rk[:3]
                    if top[0] is not None: nv += 1; v3 += d["truth"] in top; o3v += d["truth"] in rk[:3]
                n = len(ks); r["T3"] = {"n": n, "vlm_top1": 100 * t1v / n, "vlm_top3": 100 * t3v / n, "ours_top1": 100 * t1o / n, "ours_top3": 100 * t3o / n,
                                        "n_valid": nv, "vlm_top3_valid": 100 * v3 / nv if nv else None, "ours_top3_valid": 100 * o3v / nv if nv else None}
            # T4 one labelled photograph per class in the prompt vs SymPrior / SymPrior-T with the same support, same queries
            jobs = (lambda J: [j for j in J if j[0] in V.t4_subset(ds, J)])(list(V.jobs_for(ds, "T4", rows, names))) if any(k.startswith(f"T4|{ds}|") for k in D) else []
            if jobs:
                E = json.load(open(os.path.join(V.BENCH, "episodes.json")))[f"fewshot_{ds}"]["k1"]; per = {}; pool = {"y": [], "v": [], "s": [], "t": []}
                for e in range(3):
                    sup = np.array(E[e]); qk = [j[0] for j in jobs if j[0].startswith(f"T4|{ds}|e{e}|") and j[0] in D]
                    if not qk: continue
                    q = np.array([int(k.split("|")[3]) for k in qk]); pv = np.array([name2c.get(D[k]["pred"], -1) for k in qk])
                    allq = np.where(~np.isin(g, g[sup]))[0]
                    Ps = ST.posterior(Xg[sup], y[sup], Xg[allq], Tg, cls, KAPPA, 0); Pt = ST.posterior(Xg[sup], y[sup], Xg[allq], Tg, cls, KAPPA, GAMMA)
                    per[e] = {"n": int(len(q)), "vlm": ba(y[q], pv), "symprior": ba(y[q], cls[(Xg[q] @ Ps.T).argmax(1)]),
                              "symprior_t": ba(y[q], cls[(Xg[q] @ Pt.T).argmax(1)]),
                              "hybrid": ba(y[q], np.where(ST.probs(Xg[q], Pt, KAPPA).max(1) < 0.5, pv, cls[(Xg[q] @ Pt.T).argmax(1)]))}
                    pool["y"] += list(y[q]); pool["v"] += list(pv); pool["s"] += list(cls[(Xg[q] @ Ps.T).argmax(1)]); pool["t"] += list(cls[(Xg[q] @ Pt.T).argmax(1)])
                if per:
                    P_ = {k: np.array(v) for k, v in pool.items()}
                    per["pooled"] = {"n": int(len(P_["y"])), "vlm": ba(P_["y"], P_["v"]), "symprior": ba(P_["y"], P_["s"]), "symprior_t": ba(P_["y"], P_["t"]),
                                     "p_mcnemar_vlm_vs_symprior": mcnemar(P_["v"] == P_["y"], P_["s"] == P_["y"]),
                                     "p_mcnemar_vlm_vs_symprior_t": mcnemar(P_["v"] == P_["y"], P_["t"] == P_["y"]),
                                     "n_vlm_answers_invalid": int(np.sum(P_["v"] < 0))}
                    ok_ = P_["v"] >= 0
                    if ok_.sum() and not ok_.all():
                        per["pooled"].update({"vlm_valid_only": ba(P_["y"][ok_], P_["v"][ok_]), "symprior_valid_only": ba(P_["y"][ok_], P_["s"][ok_])})
                r["T4"] = per
            if r: R[m] = r
        res[ds] = R
    F.save_json(res, "fs_vlm.json")
    for ds in ("B", "C", "A"):
        for m, r in res.get(ds, {}).items():
            for t, v in r.items():
                print(ds, m, t, {k: (round(x, 1) if isinstance(x, float) else x) for k, x in (v.items() if isinstance(v, dict) else [])
                                 if not str(k).startswith("confusion") and k != "defer_curve_pct"})


if __name__ == "__main__":
    main()
