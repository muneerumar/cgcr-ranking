# Saved-results paper package

Regenerate with `python scripts/make_paper_tables.py` then `python scripts/make_paper_figures.py`. These commands perform statistical postprocessing only.

- Only saved results from the approved final run are read. No experiment, extraction, inference or dataset access occurs.
- All result confidence intervals copied from metrics.json are unchanged percentile bootstrap intervals; derived licensing/scatter/difference intervals use 10000 query resamples with seed 13.
- A [0,0] empirical bootstrap interval for zero events does not establish zero population risk.
- Main F3 uses three shared attacks as instructed; six-attack H&K is supplementary. No responsiveness data exist for the other five defense methods.
- Details-only invariance is structural because attacks change seller text, not its evidence inputs.
- GEO means the frozen Shield filter + Qwen adaptation; never full GEO-Defender or its different standalone ranker.
- The frozen grid has eight defenses/six attacks in H&K, five defenses/three attacks in each other category, and four defenses/two attacks in Llama transfer. Unrun cells are absent, never zero.
- Mimicry accepted 38/48 generations (10 failures); do not use 48 as its effectiveness denominator.
- Cost records have cache provenance; ranking latency excludes sanitizer, filter and offline extraction/verification preprocessing.

The manifest resolves exported numeric cells to the released aggregate bundle. Original upstream file hashes and calculation ancestry are retained in results/upstream_provenance.json; upstream records themselves are not included. Each numeric CSV cell references a value ID, JSON pointer and SHA-256 in the released bundle. Derived values state the calculation. Figures reference the same table cells. Source SHA-256s cover whole files. Query-bootstrap intervals are frozen publication aggregates, not recomputed from withheld records; exact binary Wilcoxon and Holm tests are independently reconstructed from released discordant counts. JSONL key paths index the array of non-empty records; text-log key paths index zero-based lines; YAML paths address its parsed mapping. No attack prose, templates, reviews or claim text is exported.

Only evaluated grid cells are exported. Human-readable LaTeX combines mean and CI columns and uses compact labels; the matching CSV preserves all precision, denominators and auxiliary fields. Dimensions, tick marks, document identifiers and style constants are layout, not empirical measurements; plotting code is hashed in the manifest.
