# Pre-registration of the follow-up experiments (written 2026-11-12, before any of them was run on fields A, B or C)

Purpose: test whether (i) open-weight vision-language chat models (VLMs) beat SymPrior on the benchmark, and (ii) whether a
principled extension of SymPrior closes or reverses any gap. To keep the results genuine, every new hyperparameter is chosen on
the SOURCE data (Okra DiseaseNet, 6 classes) only, exactly as tau was, and then fixed for fields A, B and C. Every result is
reported, including the ones that do not favour our method.

## 1. VLM baselines (src/vlm_eval.py, run locally with Ollama; qwen3-vl:8b and gemma3:4b on an 8 GB GPU)
T1 names, T2 names + descriptions, T3 naming of the unknown (top-3 from the bank minus the known classes),
T4 one labelled photo per class in the prompt (first 3 k=1 episodes; tomato: 150 fixed random queries per episode).
Options shuffled per photo with a fixed seed; temperature 0; output constrained to the option list. Fixed before results.

## 2. SymPrior-T: SymPrior with the unlabelled field photographs as extra evidence (transductive EM)
Gap-corrected embeddings as in the paper. Initialise p_c = unit(tau t_c + sum of the labelled x of class c).
Repeat 10 times: r_ic = softmax_c(kappa x_i . p_c) over the unlabelled photographs (uniform class prior);
p_c = unit(tau t_c + sum labelled x + gamma * m_c), m_c = sum_i r_ic x_i / (N_u / C)
(gamma = how many photographs' worth of evidence the unlabelled pool gives each class; invariant to the pool size).
tau = 4 (BioCLIP-2, as in the paper). Grid: kappa in {10, 20, 50, 100}, gamma in {0.5, 1, 2, 4, 8, 16}.
Selection: highest mean balanced accuracy on the source episodes (seeds 77 + 100k + d, 20 per k) averaged over k in {0, 1, 2, 5, 10}.
Test: the same episodes as the paper (seeds 1000k + d), k in {0, 1, 2, 5, 10, 20}, Wilcoxon against SymPrior and the best baseline.

## 3. One click per class (description-guided choice of WHICH photographs to label)
Budget = C clicks (one per class) or 2C. Run SymPrior-T with no labels; for each class take the unlabelled photograph with the
highest responsibility r_ic (2C budget: the top two from different groups); the oracle gives its TRUE label (which may be another
class); then run SymPrior-T with these labels. Test on all photographs outside the clicked groups.
Baselines with the same number of clicks: random photographs, one (two) per class: the paper's k = 1 / k = 2 episodes;
these are class-balanced by construction, which favours the baseline and k-means++ medoids (k = C, seed 0) on the
gap-corrected embeddings.

## 4. Hybrid with a VLM (fixed rule)
Use the VLM answer for the photographs where SymPrior-T's top posterior probability (softmax of kappa x . p) is below 0.5,
and SymPrior-T otherwise. The full accuracy-versus-deferral-fraction curve is also reported.

Anything tried beyond this document will be reported as exploratory.

## Addendum A (2026-11-12, 23:55) - results of sections 2 and 3, then one more pre-registered method
Sections 2-3 were run as registered (results/fs_symprior_t.json). Chosen on source: kappa = 50, gamma = 0.5.
SymPrior-T: small gains at k = 0 (A 88.9 -> 93.5, B 61.7 -> 65.8, C 23.5 -> 24.0) and B k = 1 (77.9 -> 79.0, p = 0.006);
no gain at k >= 2. One click per class: no better than random class-balanced labels (B budget 1: 67.7 vs 79.0). Both reported as is.

At this time only 36 VLM answers existed (qwen3-vl, field B, T1, all Cercospora photographs). Registered before further VLM results:

## 5. VLM-to-prototype distillation ("the big model teaches the small model once")
Pseudo-labels = the VLM's T2 answer (names + descriptions) for each unlabelled photograph. Variant "agree": only photographs
where the T1 and T2 answers agree. Each pseudo-labelled photograph counts as one labelled photograph (weight 1, no tuning);
prototypes p_c = unit(tau t_c + sum human-labelled x + sum pseudo-labelled x), gap-corrected, tau = 4.
(a) transductive: pseudo-labels on all photographs, accuracy on all photographs (k = 0) and on the paper's k = 1 queries;
(b) inductive (deployment): pseudo-labels on a random half of the groups (seed 0..9), accuracy on the other half, so the
distilled offline model is tested on photographs the VLM never saw.
Compared with the VLM's own T2 accuracy on the same photographs.

## Addendum B (2026-11-14, 06:50) - deviation: Qwen3-VL instruct variant
The plain Ollama tag `qwen3-vl:8b` is the *thinking* build (same digest as `8b-thinking`). With `think: false` it still returned an
empty answer for many prompts (field B: 21/252 with descriptions, 30/144 one-label prompts, 91/97 naming prompts; tomato naming
488/635), apparently after exhausting the context on hidden reasoning, and was slow. On the few naming prompts it did answer it was
much better than its overall score suggests (tomato top-3: 87% of 147 answered). A comparison dominated by empty answers would be
unfair to the model, so all Qwen3-VL tasks are re-run with the instruct build `qwen3-vl:8b-instruct` (same size and quantisation),
with exactly the same prompts, photographs, episodes and settings. Both sets of answers are released; the paper reports the
instruct build and states this deviation. No other change.
