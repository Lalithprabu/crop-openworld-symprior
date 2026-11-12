# Open-world crop leaf disease recognition in new fields: benchmark and code

Benchmark, code and per-episode results for the paper

> L. NallaPerumal and S. Shanmugasundaram, "From Unknown to Known: Open-World Crop Leaf Disease Recognition in New Fields with
> Symptom-Description Priors", ICCIDS 2027.

The task: a disease recognition system is used in a **new field**, meets **diseases absent from its training data**, and has
only **0–20 labelled photographs per disease**. It must **detect** an unknown disease, **name** it from written symptom descriptions,
and **learn** it from the few labels. **SymPrior** solves all three without training: the gap-corrected text embedding of a written
description is a conjugate von Mises–Fisher prior on each class direction, updated by every labelled photograph
(`p_c ∝ τ·t̃_c + Σ x̃_i`), with τ fixed on the source data.

## Datasets (images are not redistributed; download them from the original sources)

| Field | Source | Crop | Classes | Distinct photos | Re-shot groups |
|---|---|---|---|---|---|
| source | Okra DiseaseNet, doi:10.17632/nh7zk4hv8z.1 | okra | 6 | 293 | see `index_source.csv` |
| A | doi:10.17632/2svdj3yyrk.2 | okra | 2 | 1,082 | 1,060 |
| B | doi:10.17632/hgshn2zg3t.1 | okra | 4 (incl. yellow vein mosaic, unseen in source) | 252 | 73 |
| C | PlantDoc, github.com/pratikkayal/PlantDoc-Dataset | tomato | 8 | 726 | 711 |

All datasets are CC BY 4.0 (see their pages). Exact duplicates (MD5) are removed; near-duplicates (64-bit difference hash ≤ 6 bits)
and photographs taken within 5 s are grouped. **Labelled and test photographs never share a group.**

## Files

- `benchmark/index_*.csv` — one row per distinct photograph: file (relative to the dataset root), MD5, label, group.
  `index_source.csv` identifies Okra DiseaseNet photographs by MD5; its labels are 0 healthy, 1 Alternaria leaf spot,
  2 Cercospora leaf spot, 3 downy mildew, 4 leaf curl virus, 5 Phyllosticta leaf spot. Fields A and B use 0 healthy,
  1 Cercospora leaf spot, 2 leaf curl, 3 yellow vein mosaic; field C (tomato) uses the order in `index_C.csv`.
- `benchmark/episodes.json` — every evaluation episode (support indices). Queries = all photographs outside the support groups.
- `descriptions/descriptions.json` — the symptom descriptions and the naming banks (okra 15 entries, tomato 16).
- `code/` — `fewshot.py` (methods, episodes, baselines incl. Tip-Adapter, Align+Mix, gap-corrected prototype/probe, MCM, and the
  fixed-weight ablation), `analysis.py`, `make_tex.py`,
  `tomato.py`, `prepare_tomato.py`, `export_*.py`.
- `results/` — all results, including one record per episode (`*_raw.json`), so new methods can be compared with paired tests.

## Reproducing

```
python prepare_tomato.py <PlantDoc folder> <out>      # field C index (optional; index provided)
python fewshot.py embed <ext_a> <ext_b>               # okra embeddings (needs GPU/CPU + open_clip, timm)
python tomato.py embed                                 # tomato embeddings
python fewshot.py probe && python fewshot.py openset   # all experiments (seconds to minutes on a CPU)
python analysis.py && python make_tex.py               # every number, table and figure of the paper
```

## Licence

Code: MIT (`LICENSE`). Indices, episodes and descriptions: CC BY 4.0. Please cite the paper (`CITATION.cff`) and the
original datasets.
