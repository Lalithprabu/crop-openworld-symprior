"""Writes manuscript/numbers.tex (every number quoted in the text, as \\N{key}) and the three tables, from the result files."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); MS = os.path.join(ROOT, "manuscript")
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
P, FT, OS, N = R("fs_probe.json"), R("fs_finetune.json"), R("fs_openset.json"), R("paper_numbers.json")
f1 = lambda x: f"{x:.1f}"
sg = lambda x: f"{x:+.1f}".replace("-", "$-$")                      # signed difference with a typographic minus
pf = lambda p: ("<0.001" if p < 0.001 else f"={p:.3f}" if p < 0.01 else f"={p:.2f}")      # includes the operator
M = {}

# datasets
for ds in ("A", "B", "C"):
    i = N["info"][ds]; M[f"n{ds}"] = f"{i['n_images']:,}"; M[f"g{ds}"] = f"{i['n_groups']:,}"; M[f"maxk{ds}"] = str(max(int(k.rsplit('|', 1)[1]) for k in P[ds] if '|' in k))
M["tauBCtwo"], M["tauBC"] = str(N["tau"]["bioclip2"]), str(N["tau"]["bioclip"])
M["rawZS"], M["corrZS"] = f1(N["fig1_raw_zeroshot_balacc"]), f1(N["fig1_corrected_zeroshot_balacc"])
# few-shot
for ds in ("A", "B", "C"):
    for k in (0, 1, 2, 5, 10, 20, 40):
        r = N.get(f"{ds}|k{k}")
        if not r:
            continue
        p = f"{ds}k{k}"
        M[p + "sym"] = f1(r["symprior"][0]); M[p + "symlo"] = f1(r["symprior"][1]); M[p + "symhi"] = f1(r["symprior"][2])
        M[p + "best"] = f1(r["best_baseline_mean"]); M[p + "ind"] = f1(r["inductive"]); M[p + "zs"] = f1(r["zeroshot"]); M[p + "nogap"] = f1(r["nogap"])
        if k > 0:
            M[p + "diff"] = sg(r['diff_mean']); M[p + "difflo"] = sg(r['diff_ci'][0]); M[p + "diffhi"] = sg(r['diff_ci'][1])
            M[p + "p"] = pf(r["p_wilcoxon"]); M[p + "wins"] = str(r["wins"]); M[p + "n"] = str(r["n"])
            M[p + "same"] = sg(r['vs_same_probe']); M[p + "samep"] = pf(r["vs_same_probe_p"]); M[p + "samewins"] = str(r["vs_same_probe_wins"])
            M[p + "gapgain"] = sg(r['gap_gain']); M[p + "gapp"] = pf(r["gap_p"])
            M[p + "indvsbest"] = sg(r['inductive_vs_best']); M[p + "indvsbestp"] = pf(r["inductive_vs_best_p"])
        if "yvmv_recall_symprior" in r:
            M[p + "yvmv"] = f1(r["yvmv_recall_symprior"])
for ds in ("A", "B"):
    for k, v in N["finetune"][ds].items():
        key = k.replace("mnv3_ft_", "").replace("|ft|", "k").replace("_", "")
        M[f"{ds}ft{key}"] = f1(v[0])
# open world
o = OS
for k in (1, 2, 5):
    s = o.get(f"source_loco|bioclip2|k{k}")
    if s:
        for m in ("symprior", "centroid", "knn", "msp"):
            M[f"lococ{k}{m}"] = f1(s[f"auroc_{m}"][0])
        M[f"loco{k}nametop1"] = f1(s["name_group_top1"][0]); M[f"loco{k}nametop3"] = f1(s["name_group_top3"][0]); M[f"loco{k}nameimg"] = f1(s["name_img_top1"][0])
        for m in ("centroid", "msp", "knn"):
            M[f"loco{k}p{m}"] = pf(N["open"][f"source_loco|bioclip2|k{k}"][f"symprior_vs_{m}_p"])
for k in (1, 2):
    s = o.get(f"fieldB|k{k}")
    if s:
        for m in ("symprior", "centroid", "knn", "msp"):
            M[f"fb{k}{m}"] = f1(s[f"clip:bioclip2|auroc_{m}"][0]); M[f"fb{k}closed{m}"] = f1(s[f"clip:bioclip2|closed_bal_acc_{m}"][0])
        M[f"fb{k}bcsym"] = f1(s["clip:bioclip|auroc_symprior"][0])
        M[f"fb{k}nameimg"] = f1(s["clip:bioclip2|name_img_top1"][0]); M[f"fb{k}nametop1"] = f1(s["clip:bioclip2|name_group_top1"][0])
        M[f"fb{k}nametop3"] = f1(s["clip:bioclip2|name_group_top3"][0])
        M[f"fb{k}nametopcount"] = str(int(round(s["clip:bioclip2|name_group_top1"][0] / 100 * 20)))
        M[f"fb{k}pcentroid"] = pf(N["open"][f"fieldB|k{k}"]["symprior_vs_centroid_p"]); M[f"fb{k}pmsp"] = pf(N["open"][f"fieldB|k{k}"]["symprior_vs_msp_p"]); M[f"fb{k}pknn"] = pf(N["open"][f"fieldB|k{k}"]["symprior_vs_knn_p"])
        best_other = max((s[f"{f}|auroc_{m}"][0], f, m) for f in ("bioclip", "dinov2", "mnv3", "source") for m in ("centroid", "knn", "msp"))
        M[f"fb{k}bestimg"] = f1(best_other[0])

# extra quantities quoted in the text
M["Ak0zsBC"], M["Bk0zsBC"] = f1(P["A"]["clip_bioclip|zeroshot|0"]["bal_acc"][0]), f1(P["B"]["clip_bioclip|zeroshot|0"]["bal_acc"][0])
M["Bk1ftbest"] = f1(max(FT["B"][f"mnv3_ft_{i}|ft|1"]["bal_acc"][0] for i in ("imagenet", "source")))
M["AsymBCtwok"] = f1(P["A"]["clip_bioclip|symprior|2"]["bal_acc"][0]); M["AsymBCfortyk"] = f1(P["A"]["clip_bioclip|symprior|40"]["bal_acc"][0])
M["nbaselines"] = str(len([k for k in P["B"] if k.endswith("|1") and k.split("|")[1] in ("probe", "centroid", "tipadapter", "tipadapter_gap", "alignmix", "centroid_gap", "probe_gap", "zeroshot")]))
gaps = [N[f"{ds}|k{k}"]["symprior"][0] - N[f"{ds}|k{k}"]["inductive"] for ds in ("A", "B") for k in (0, 1, 2, 5, 10, 20, 40) if f"{ds}|k{k}" in N]
M["indgapmax"] = f1(max(gaps))
M["locomaxp"] = pf(max(N["open"][f"source_loco|bioclip2|k{k}"][f"symprior_vs_{m}_p"] for k in (1, 2, 5) for m in ("centroid", "msp", "knn")))

# reviewer additions: gap-corrected Tip-Adapter, operating point, realistic naming, pool sensitivity, BioCLIP
for ds in ("A", "B", "C"):
    for k in (1, 2, 5, 10):
        if f"clip_bioclip2|tipadapter_gap|{k}" in P[ds]:
            M[f"{ds}k{k}tipgap"] = f1(P[ds][f"clip_bioclip2|tipadapter_gap|{k}"]["bal_acc"][0])
            M[f"{ds}k{k}tipgaptuned"] = f1(P[ds][f"clip_bioclip2|tipadapter_gap_tuned|{k}"]["bal_acc"][0])
# Align+Mix (Goswami et al.), lambda chosen on the source episodes like tau
M["lamAM"] = str(P["_lambda"]["bioclip2"]["chosen"])
for ds in ("A", "B", "C"):
    for k in (1, 2, 5, 10):
        if f"clip_bioclip2|alignmix|{k}" in P[ds]:
            M[f"{ds}k{k}am"] = f1(P[ds][f"clip_bioclip2|alignmix|{k}"]["bal_acc"][0]); M[f"{ds}k{k}amtuned"] = f1(P[ds][f"clip_bioclip2|alignmix_tuned|{k}"]["bal_acc"][0])
M["Bamtunedmax"] = f1(max(P["B"][f"clip_bioclip2|alignmix_tuned|{k}"]["bal_acc"][0] for k in (1, 2)))
M["Ctipgapmax"] = f1(max(P["C"][f"clip_bioclip2|tipadapter_gap|{k}"]["bal_acc"][0] for k in (1, 2, 5, 10, 20)))
M["Cammax"] = f1(max(P["C"][f"clip_bioclip2|alignmix|{k}"]["bal_acc"][0] for k in (1, 2, 5, 10, 20)))
# field C (tomato): zero-shot with BioCLIP, best probe at k=20, transferred tau vs best tau
M["Ck0zsBC"] = f1(P["C"]["clip_bioclip|zeroshot|0"]["bal_acc"][0])
M["CprobebestTwenty"] = f1(max(v["bal_acc"][0] for k_, v in P["C"].items() if k_.endswith("|probe|20")))
TS = P["_tau_sensitivity_C"]
M["Ctaugapmax"] = f1(max(max(TS[f"k{k}|tau{t}"] for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32)) - TS[f"k{k}|tau4"] for k in (1, 2, 5)))
M["Ctaubest"] = str(max((0.25, 0.5, 1, 2, 4, 8, 16, 32), key=lambda t: sum(TS[f"k{k}|tau{t}"] for k in (1, 2, 5)))).replace(".0", "")
for k in (1, 2, 5):
    s_ = o[f"fieldC_loco|bioclip2|k{k}"]
    for m in ("symprior", "centroid", "knn", "msp"):
        M[f"Cloco{k}{m}"] = f1(s_[f"auroc_{m}"][0]); M[f"Cloco{k}fpr{m}"] = f"{s_[f'fpr95_{m}'][0]:.0f}"
    M[f"Cloco{k}nametop1"] = f1(s_['name_group_top1'][0]); M[f"Cloco{k}nametop3"] = f1(s_['name_group_top3'][0])
    M[f"Cloco{k}nameimg"] = f1(s_["name_img_top1"][0])
M["locomcmpone"] = pf(N["open"]["source_loco|bioclip2|k1"]["symprior_vs_mcm_p"])
M["lamFix"] = str(P["_lambda_fixed_mix"]["bioclip2"]["chosen"])
for ds in ("A", "B", "C"):
    for k in (1, 2, 5, 10, 20):
        r = N.get(f"{ds}|k{k}")
        if r and "vs_mixfix" in r:
            M[f"{ds}k{k}mix"] = f1(P[ds][f"clip_bioclip2|mixfix_gap|{k}"]["bal_acc"][0]); M[f"{ds}k{k}mixp"] = pf(r["vs_mixfix_p"])
            M[f"{ds}k{k}mixdiff"] = sg(r["vs_mixfix"])
            M[f"{ds}k{k}protogap"] = f1(P[ds][f"clip_bioclip2|centroid_gap|{k}"]["bal_acc"][0])
            M[f"{ds}k{k}probegap"] = f1(P[ds][f"clip_bioclip2|probe_gap|{k}"]["bal_acc"][0])
M["Cmaxall"] = f1(max(v["bal_acc"][0] for k_, v in P["C"].items() if "|" in k_ and "tuned" not in k_))
for key, nm in (("source_loco|bioclip2|k5", "lococ5mcm"), ("fieldC_loco|bioclip2|k5", "Cloco5mcm")):
    M[nm] = f1(o[key]["auroc_mcm"][0])
M["fb2mcm"], M["fb1mcm"] = f1(o["fieldB|k2"]["clip:bioclip2|auroc_mcm"][0]), f1(o["fieldB|k1"]["clip:bioclip2|auroc_mcm"][0])
M["Clocop"] = pf(max(N["open"]["fieldC_loco|bioclip2|k5"][f"symprior_vs_{m}_p"] for m in ("centroid", "msp", "knn")))
M["Btipgaptunedmax"] = f1(max(P["B"][f"clip_bioclip2|tipadapter_gap_tuned|{k}"]["bal_acc"][0] for k in (1, 2)))
for k in (1, 2, 5):
    s = o[f"source_loco|bioclip2|k{k}"]
    for m in ("symprior", "centroid", "knn", "msp"):
        M[f"lococ{k}fpr{m}"] = f"{s[f'fpr95_{m}'][0]:.0f}"
for k in (1, 2):
    s = o[f"fieldB|k{k}"]; pre = "clip:bioclip2|"
    for m in ("symprior", "centroid", "knn", "msp"):
        M[f"fb{k}fpr{m}"] = f"{s[pre + f'fpr95_{m}'][0]:.0f}"
    for fr in (10, 20, 40):
        M[f"fb{k}flag{fr}prec"] = f"{s[pre + f'flag{fr}_precision'][0]:.0f}"
        M[f"fb{k}flag{fr}top1"] = f"{s[pre + f'flag{fr}_name_top1'][0]:.0f}"
        M[f"fb{k}flag{fr}top3"] = f"{s[pre + f'flag{fr}_name_top3'][0]:.0f}"
    M[f"fb{k}poolnoU"] = f1(s[pre + "auroc_symprior_poolnoU"][0])
    M[f"fb{k}nametopthree"] = f"{s[pre + 'name_group_top3'][0]:.0f}"
    M[f"fb{k}nameoneint"] = f"{s[pre + 'name_group_top1'][0]:.0f}"
M["fbonebcsym"], M["fbtwobcsym"] = f1(o["fieldB|k1"]["clip:bioclip|auroc_symprior"][0]), f1(o["fieldB|k2"]["clip:bioclip|auroc_symprior"][0])
lc = o["source_loco|bioclip2|k5"]
for nm, key in (("H", "Healthy"), ("AL", "Alternaria leaf spot"), ("CL", "Cercospora leaf spot"), ("DM", "Downy mildew"), ("LC", "Leaf curl"), ("PH", "Phyllosticta leaf spot")):
    M[f"loconame{nm}"] = str(int(round(lc[f"per_class_name_group_{key}"][0] / 10)))          # of 10 episodes

# re-shot groups in field B (from the leakage-safe grouping)
Zb = np.load(os.path.join(ROOT, "cache", "fs_B.npz")); yb, gb = Zb["y"], Zb["g"]
for c, nm in enumerate(["H", "CLS", "LC", "YVMV"]):
    gs, cnt = np.unique(gb[yb == c], return_counts=True)
    M[f"B{nm}n"], M[f"B{nm}g"], M[f"B{nm}max"] = str(int((yb == c).sum())), str(len(gs)), str(int(cnt.max()))


# vision-language chat models (results/fs_vlm.json, src/vlm_analyze.py) and SymPrior-T (results/fs_symprior_t.json)
V_, T_ = R("fs_vlm.json"), R("fs_symprior_t.json")
MOD = {"qw": "qwen3-vl_8b-instruct", "ge": "gemma3_4b"}   # the thinking build (qwen3-vl_8b) is supplementary
for ds in ("B", "C"):
    for a, m in MOD.items():
        r = V_.get(ds, {}).get(m, {})
        for t in ("T1", "T2"):
            if t in r: M[f"v{ds}{a}{t}"] = f1(r[t]["vlm"])
        if "T2" in r:
            M[f"v{ds}{a}p"] = pf(max(r["T2"]["p_mcnemar_vlm_vs_symprior"], r["T1"]["p_mcnemar_vlm_vs_symprior"]))
            for var in ("all",):
                M[f"v{ds}{a}distil"] = f1(r["T2"][f"distil_{var}_transductive"]); M[f"v{ds}{a}distilind"] = f1(r["T2"][f"distil_{var}_inductive"][0])
                M[f"v{ds}{a}half"] = f1(r["T2"]["vlm_on_inductive_halves"]); M[f"v{ds}{a}pseudo"] = f1(r["T2"]["distil_all_pseudo_acc"])
                M[f"v{ds}{a}distilagree"] = f1(r["T2"]["distil_agree_transductive"])
                M[f"v{ds}{a}distilp"] = pf(r["T2"]["distil_all_p_vs_symprior"]); M[f"v{ds}{a}distilpv"] = pf(r["T2"]["distil_all_p_vs_vlm"])
        pl = r.get("T4", {}).get("pooled")
        if pl and pl["n"] >= 0.9 * {"B": 144, "C": 96}[ds]:
            M[f"v{ds}{a}four"] = f1(pl["vlm"]); M[f"s{ds}four"] = f1(pl["symprior"]); M[f"t{ds}four"] = f1(pl["symprior_t"])
            M[f"v{ds}{a}fourp"] = pf(pl["p_mcnemar_vlm_vs_symprior"]); M[f"v{ds}{a}fourn"] = str(pl["n"])
            if "vlm_valid_only" in pl:
                M[f"v{ds}{a}fourvalid"] = f1(pl["vlm_valid_only"]); M[f"v{ds}{a}fourinvalid"] = str(pl["n_vlm_answers_invalid"])
        if "T3" in r:
            M[f"v{ds}{a}nt1"], M[f"v{ds}{a}nt3"] = f1(r["T3"]["vlm_top1"]), f1(r["T3"]["vlm_top3"])
            M[f"o{ds}{a}nt1"], M[f"o{ds}{a}nt3"], M[f"v{ds}{a}nn"] = f1(r["T3"]["ours_top1"]), f1(r["T3"]["ours_top3"]), str(r["T3"]["n"])
M["stKappa"], M["stGamma"] = str(T_["_chosen"]["kappa"]), str(T_["_chosen"]["gamma"])
for ds in ("A", "B", "C"):
    for k in ("k0", "k1", "k2"):
        e = T_["fewshot"][ds].get(k)
        if e:
            M[f"st{ds}{k}"] = f1(e["symprior_t"][0])
            if "p_vs_symprior" in e: M[f"st{ds}{k}p"] = pf(e["p_vs_symprior"])

with open(os.path.join(MS, "numbers.tex"), "w") as f:
    f.write("% generated by src/make_tex.py from the result files - do not edit\n")
    for k, v in M.items():
        f.write(f"\\expandafter\\def\\csname n@{k}\\endcsname{{{v}}}\n")
    f.write("\\makeatletter\\newcommand{\\N}[1]{\\@ifundefined{n@#1}{\\textbf{??#1}}{\\csname n@#1\\endcsname}}\\makeatother\n")

# Table II: few-shot balanced accuracy
KA, KB = [0, 1, 2, 5, 10, 20, 40], [0, 1, 2]
rows = [("Zero-shot text", "clip_bioclip2|zeroshot", "BioCLIP-2"), 
        ("Tip-Adapter~\\cite{zhang2022tip}", "clip_bioclip2|tipadapter", "BioCLIP-2"), 
        ("Tip-Adapter, gap corr.", "clip_bioclip2|tipadapter_gap", "BioCLIP-2"),
        ("Align+Mix~\\cite{goswami2026alignmix}", "clip_bioclip2|alignmix", "BioCLIP-2"),
        ("Prototype~\\cite{snell2017prototypical}", "clip_bioclip2|centroid", "BioCLIP-2"),
        ("Prototype, gap corr.", "clip_bioclip2|centroid_gap", "BioCLIP-2"),
        ("Linear probe", "clip_bioclip2|probe", "BioCLIP-2"), ("Linear probe, gap corr.", "clip_bioclip2|probe_gap", "BioCLIP-2"), ("Linear probe", "clip_bioclip|probe", "BioCLIP"),
        ("Linear probe", "source|probe", "MNv3-src"),
        ("Fine-tuning", "FT:imagenet", "MNv3-IN"), ("Fine-tuning", "FT:source", "MNv3-src"),
        ("SymPrior, no gap corr.", "clip_bioclip2|symprior_nogap", "BioCLIP-2"),
        ("Fixed-weight mix, gap corr.", "clip_bioclip2|mixfix_gap", "BioCLIP-2"),
        ("SymPrior, inductive", "clip_bioclip2|symprior_inductive", "BioCLIP-2"),
        ("\\textbf{SymPrior (ours)}", "clip_bioclip2|symprior", "BioCLIP-2"), ("\\textbf{SymPrior (ours)}", "clip_bioclip|symprior", "BioCLIP")]


def val(ds, key, k):
    if key.startswith("FT:"):
        v = FT[ds].get(f"mnv3_ft_{key[3:]}|ft|{k}")
    else:
        v = P[ds].get(f"{key}|{k}")
    return None if v is None else v["bal_acc"]


best = {(ds, k): max(v[0] for v in (val(ds, key, k) for _, key, _ in rows) if v) for ds, KK in (("A", KA), ("B", KB)) for k in KK}
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Balanced accuracy (\\%) in the two okra fields with $k$ labelled photographs per class: mean over 20 group-disjoint "
         "episodes (fine-tuning: 5), standard deviation in small type, best value per column in bold. MNv3-IN/-src: MobileNetV3-Large with ImageNet "
         "weights or trained on Okra DiseaseNet; a dash marks a method that needs labels; field B allows $k\\le2$ (Section~\\ref{sec:setup}). "
         "Probes on DINOv2 and ImageNet MobileNetV3, the tomato results (Fig.~\\ref{fig:curves}) and all per-episode results are in the repository.}",
         "\\label{tab:fewshot}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
         "\\begin{tabular}{ll" + "c" * len(KA) + "|" + "c" * len(KB) + "}", "\\toprule",
         "& & \\multicolumn{%d}{c|}{Field A (healthy, Cercospora; chance 50\\%%)} & \\multicolumn{%d}{c}{Field B (4 classes; chance 25\\%%)}\\\\" % (len(KA), len(KB)),
         "Method & Encoder & " + " & ".join(f"$k$={k}" for k in KA) + " & " + " & ".join(f"$k$={k}" for k in KB) + "\\\\", "\\midrule"]
for name, key, enc in rows:
    cells = []
    for ds, KK in (("A", KA), ("B", KB)):
        for k in KK:
            v = val(ds, key, k)
            if v is None:
                cells.append("--"); continue
            s = f"{v[0]:.1f}"
            if abs(v[0] - best[(ds, k)]) < 0.05:
                s = f"\\textbf{{{s}}}"
            if v[1] > 0:
                s += f"\\,{{\\tiny{v[1]:.1f}}}"
            cells.append(s)
    if key == "clip_bioclip2|symprior_nogap":
        lines.append("\\midrule")
    lines.append(f"{name} & {enc} & " + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(os.path.join(MS, "tab_fewshot.tex"), "w").write("\n".join(lines))

# Table III: detection AUROC and closed-set accuracy
meths = [("symprior", "\\textbf{SymPrior (ours)}"), ("mcm", "MCM~\\cite{ming2022mcm}"), ("centroid", "Prototype"), ("knn", "Nearest neighbour~\\cite{sun2022knn}"),
         ("msp", "Max.\\ softmax prob.~\\cite{hendrycks2017baseline}")]
L = ["\\begin{table}[t]", "\\centering",
     "\\caption{Detecting an unknown disease (AUROC, \\%), BioCLIP-2 "
     "embeddings. DiseaseNet and tomato: each of the six or eight classes held out in turn (60 or 80 episodes per $k$). Field B: "
     "yellow vein mosaic unknown, healthy, Cercospora and leaf curl known (20 episodes).}",
     "\\label{tab:open}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.2pt}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lccc|cc|ccc}", "\\toprule",
     "& \\multicolumn{3}{c|}{Okra DiseaseNet} & \\multicolumn{2}{c|}{Okra field B} & \\multicolumn{3}{c}{Tomato}\\\\",
     "Score & $k$=1 & $k$=2 & $k$=5 & $k$=1 & $k$=2 & $k$=1 & $k$=2 & $k$=5\\\\", "\\midrule"]
cols = [("source_loco|bioclip2|k1", "auroc_{}"), ("source_loco|bioclip2|k2", "auroc_{}"), ("source_loco|bioclip2|k5", "auroc_{}"),
        ("fieldB|k1", "clip:bioclip2|auroc_{}"), ("fieldB|k2", "clip:bioclip2|auroc_{}"),
        ("fieldC_loco|bioclip2|k1", "auroc_{}"), ("fieldC_loco|bioclip2|k2", "auroc_{}"), ("fieldC_loco|bioclip2|k5", "auroc_{}")]
bestc = [max(o[g][t.format(m)][0] for m, _ in meths) for g, t in cols]
for m, nm in meths:
    cells = []
    for (g, t), b in zip(cols, bestc):
        v = o[g][t.format(m)][0]; s = f"{v:.1f}"
        cells.append(f"\\textbf{{{s}}}" if abs(v - b) < 0.05 else s)
    L.append(nm + " & " + " & ".join(cells) + "\\\\")
L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open(os.path.join(MS, "tab_open.tex"), "w").write("\n".join(L))

g = lambda k: M.get(k, "--")
T = ["\\begin{table}[t]", "\\centering",
     "\\caption{Chat models against SymPrior on the same photographs (balanced accuracy; naming: unseen disease in the top three, per photograph; \\%). "
     "1 label: one labelled photograph per class in the prompt, " + g("vBqwfourn") + " (okra) and 96 (tomato) queries."
     + ("" if all(k in M for k in ("vCqwfour", "vCgent3")) else " --: not available.")
     + ("" if V_["C"].get(MOD["qw"], {}).get("T3", {}).get("n", 726) == 726 else " Tomato naming on the %d photographs answered by Qwen3-VL." % V_["C"][MOD["qw"]]["T3"]["n"]) + "}",
     "\\label{tab:vlm}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.4pt}", "\\resizebox{\\columnwidth}{!}{%",
     "\\begin{tabular}{lcccc|cccc}", "\\toprule",
     "& \\multicolumn{4}{c|}{Okra field B} & \\multicolumn{4}{c}{Tomato}\\\\",
     "& \\multicolumn{2}{c}{0 labels} & 1 label & Name & \\multicolumn{2}{c}{0 labels} & 1 label & Name\\\\",
     "Method & names & +desc. & & top-3 & names & +desc. & & top-3\\\\", "\\midrule",
     "Qwen3-VL 8B~\\cite{qwen3vl} & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (g("vBqwT1"), g("vBqwT2"), g("vBqwfour"), g("vBqwnt3"), g("vCqwT1"), g("vCqwT2"), g("vCqwfour"), g("vCqwnt3")),
     "Gemma 3 4B~\\cite{gemma3} & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (g("vBgeT1"), g("vBgeT2"), g("vBgefour"), g("vBgent3"), g("vCgeT1"), g("vCgeT2"), g("vCgefour"), g("vCgent3")),
     "\\midrule",
     "\\textbf{SymPrior (ours)} & -- & %s & %s & %s & -- & %s & %s & %s\\\\" % (g("Bk0sym"), g("sBfour"), g("oBqwnt3"), g("Ck0sym"), g("sCfour"), g("oCqwnt3")),
     "\\textbf{SymPrior-T (ours)} & -- & %s & %s & -- & -- & %s & %s & --\\\\" % (g("stBk0"), g("tBfour"), g("stCk0"), g("tCfour")),
     "\\textbf{SymPrior + Qwen3-VL answers} & -- & \\textbf{%s} & -- & -- & -- & %s & -- & --\\\\" % (g("vBqwdistil"), g("vCqwdistil")),
     "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open(os.path.join(MS, "tab_vlm.tex"), "w").write("\n".join(T))
print(len(M), "numbers written")
