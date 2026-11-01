# CGCR: Claim-Grounded Counterfactual Ranking

CGCR extracts typed product claims, verifies them against a fixed catalog and
customer-review evidence, and deterministically renders licensed claims for an
LLM ranker. This release includes the public library, frozen settings, unit tests,
and aggregate results needed to reproduce every distributed table and figure.

Authors: **Muhammad Muneer Umar** (corresponding,
muneer.umar@kust.edu.pk; [ORCID](https://orcid.org/0000-0003-2459-6470)) and
**Salma Perveen** (salmaperveen89@gmail.com), Institute of Computing, Kohat
University of Science & Technology, Kohat 26000, Khyber Pakhtunkhwa, Pakistan.
See `CITATION.cff` for citation metadata. Code license: MIT; see `DATA.md` for
separate upstream licenses.

## Reproducible installation

The lock is the measured **Python 3.11 / macOS arm64** environment, including MLX
and the CPU verifier. GPU support is omitted. Install [uv](https://docs.astral.sh/uv/)
and run:

```sh
make setup
make test
```

`setup` creates a fresh virtual environment, installs every exact lock pin and
installs the library without resolving additional dependencies. Other platforms
may render tables with a separately resolved scientific Python environment, but
the distributed Mac lock is the reproducibility reference.

## Regenerate the publication artifacts

For the standalone TikZ pipeline install `brew install tectonic poppler` first.
The export uses Tectonic 0.17.0 in deterministic mode and Poppler 26.05.0 at
300 dpi. The compiler fetches its TeX resources on first use; later builds reuse
its cache. F1 is exported as PDF/PNG as well as editable TikZ source.


No data, model downloads or inference are needed:

```sh
make tables     # all T1–T6 tables, paired tests and supplementary CSV/LaTeX
make figures    # all F2–F5 and supplementary vector PDF / 300-dpi PNG; F1 TikZ
```

For an individual table or figure use the same complete-generation command in
the mapping below. Each invocation rebuilds its complete output family so shared
provenance stays consistent.

| Artifact | One command |
|---|---|
| T1 robustness + Wilcoxon/Holm | `make tables` |
| T2 clean NDCG | `make tables` |
| T3 responsiveness | `make tables` |
| T4 Llama transfer | `make tables` |
| T5 adaptive flooding/mimicry | `make tables` |
| T6 online/offline cost | `make tables` |
| F1 pipeline (TikZ source) | `make figures` |
| F2 strongest attack | `make figures` |
| F3 robustness/responsiveness, common three attacks | `make figures` |
| F4 clean quality | `make figures` |
| F5 flooding | `make figures` |
| H&K six-attack supplementary table/figure | `make tables figures` |

Outputs live under `paper/tables/` and `paper/figures/`.
`paper/results_manifest.json` maps every empirical numeric cell and plotted point
to `results/publication_metrics.json`, a JSON pointer and SHA-256. The numeric
bundle preserves the saved publication values and bootstrap intervals exactly.
It contains no query-level vectors. Exact binary Wilcoxon and Holm values are
checked again from aggregate discordant counts during regeneration. The hashed
upstream calculation graph is in `results/upstream_provenance.json`; it references
unbundled research records rather than claiming they are distributed.

Main F3 uses core_reasoning_opt5, core_reasoning and framing_M3_opt5 in every
category; H&K's six-attack view is supplementary. Unrun cells are absent. Promote@3
is unconditional on paired accepted cases; NRG excludes clean-rank-one cases;
quality uses utility-eligible sets. Structural invariance of details-only follows
from its unchanged input. Zero-event bootstrap intervals do not prove zero risk.
Shield filter + Qwen is an adaptation, not full GEO-Defender. Recorded logical
call cost retains source latency on cache reuse and excludes preprocessing.

## Download data and prepare candidates

Read `DATA.md` and establish the necessary upstream permissions first.
`configs/data_sources.json` contains URL, expected byte length and the SHA-256
of each original study download. The download command is a dry plan by default:

```sh
make data
make data DATA_ARGS=--download  # explicit network/disk operation; validates hashes
.venv/bin/python scripts/prepare_data.py --category Home_and_Kitchen
.venv/bin/python scripts/prepare_evidence.py --category Home_and_Kitchen
# Repeat the two preparation commands for Electronics and Tools_and_Home_Improvement.
```

Files go under ignored `data/`. Gzip readers stream line by line; large raw files
are never loaded wholesale. A hash mismatch stops processing. Downloads resume
from bounded byte ranges; candidate selection and query splits use seed 13.
The raw-data preparation is optional for reproducing the released tables/figures.

## Download and convert models

Pinned official model repositories and revisions are in `configs/final_study.yaml`.
Use the configured Qwen and Llama **4-bit, group size 64, affine** conversions.
For gated Llama, accept the model terms and run `hf auth login` locally first.

```sh
.venv/bin/python scripts/convert_models.py --model-key qwen_4bit
.venv/bin/python scripts/convert_models.py --model-key llama_4bit
.venv/bin/hf download cross-encoder/nli-deberta-v3-base --revision 6c749ce3425cd33b46d187e45b92bbf96ee12ec7 --local-dir models/nli-deberta-v3-base
```

Conversions call `mlx_lm.convert` on pinned official weights, verify source shard
hashes, and store quantization metadata plus output hashes under ignored `models/`.
Load one LLM per process. The verifier uses CPU, four threads, trimmed_mean and
tau=0.4, calibrated on dev only. Model weights and restricted attack generators
are not bundled. See `DISCLOSURE.md` for research-material requests. The public
copy supports aggregate reproduction and the method components; a new full attack
study additionally needs the restricted bundle and upstream datasets.

## Measured runtime on Apple M-series

The completed study used a 16 GB Apple M5 MacBook Air. From the saved resource
logs (`results/runtime.json`), total supervised time across recorded phase attempts
was **99.70 hours**. It includes interrupted attempts and model startup, and
excludes idle gaps without a worker; it is not a guarantee for another machine.

| Study stage | Supervised hours |
|---|---:|
| electronics_qwen | 26.07 |
| hk_llama_transfer | 4.08 |
| hk_qwen_full | 42.42 |
| responsiveness | 2.39 |
| tools_qwen | 24.75 |

Per-call latency/tokens and offline extraction/verification per 1,000 indexed
items are in `paper/tables/T6a_ranking_cost.csv` and `T6b_offline_cost.csv`.
The original five-chart publication build took 1.82 s for tables and 2.33 s
for figures (warm font cache, excluding installation), as recorded in
`results/publication_build_cost.json`. Those historical timings predate the
standalone F1 export; its first build also fetches TeX resources. Download/install time is separate. For new long computations,
use a detached terminal multiplexer and `caffeinate -imsu`, save resumable
checkpoints, keep at least 1.5 GB available for MLX and 2.5 GB for CPU work, and
log resident and Metal memory separately. Source claims and the frozen catalog
snapshot remain assumptions; the code does not establish that reviews are true.
