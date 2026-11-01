# Data, model and third-party provenance

No datasets, customer reviews, model weights, modified listings or synthetic
reviews are redistributed. `configs/data_sources.json` records the exact URLs,
compressed byte lengths and SHA-256 digests of the completed data downloads used
in this study. Checksums identify those snapshots; changed upstream files fail
validation rather than silently replacing the inputs.

| Source | Recorded license / terms | Distribution policy |
|---|---|---|
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) / [dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | No license stated in the metadata checked for the study | Download directly only after establishing permission; no data included |
| [Amazon ESCI Shopping Queries](https://github.com/amazon-science/esci-data) | Repository Apache-2.0; consult upstream notices and data terms | URLs and checksums only |
| [FramingBench code](https://github.com/m1k2zoo/framingbench) | MIT | Public sanitizer prompt with its original license notice in configs/prompts/; attack material withheld |
| [FramingBench dataset](https://huggingface.co/datasets/framingbench/framingbench) | CC-BY-4.0 | Not included |
| [CORE / ProductBench](https://github.com/haibojin001/AmazonCOREBench) | No repository license stated in the study audit | Link only; local evaluation, no redistribution |
| [GEO-Bench ranking benchmark](https://github.com/glad-lab/geobench) | No repository license stated in the study audit | Link only; no redistribution |
| [GEO-Defender](https://github.com/Ccchi5ato/GEO-Defender) | MIT code | No upstream code or weights bundled; filter interface retained |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Apache-2.0 | Pinned repository revision, download separately |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Llama 3.1 Community License, gated access | Accept terms and authenticate locally; no weights included |
| [NLI DeBERTa-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | Apache-2.0 | Pinned revision, CPU inference |

License statements reproduce the study's recorded metadata audit, rather than
claiming new rights. Re-check upstream terms before downloading or redistributing.
The MIT license at the repository root covers our code; third-party notices and
model/data terms remain separate. Tiny test fixtures are synthetic examples,
except the catalog examples used to test unit and neutral-title handling.

The categories are Home_and_Kitchen, Electronics and Tools_and_Home_Improvement.
ESCI examples are filtered to US and small_version=1, with ten distinct parent
ASINs per candidate set. Child ASINs are joined through the review mapping;
variant-sensitive mismatches remain unknown. Catalog features/title/description
are seller text; only typed details, price/rating/category and customer-review
evidence can support neutral renderings. The catalog snapshot is fixed at attack
time, an experimental assumption rather than independent authentication.

The release's frozen aggregate bundle reproduces the publication tables and
figures without dataset access. Upstream source hashes are retained as provenance
references only. Query identifiers, query-level vectors and generated content
are removed from exported metrics. Statistical intervals were computed on the
saved query-level records before release; they are not a fresh analysis of raw
records here. Exact binary paired tests can be recomputed from discordant counts.
