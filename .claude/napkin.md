# Napkin

## 2026-03-16

- Repo had no existing `.claude/napkin.md`; created one to satisfy the per-repo tracking workflow.
- `train_embeddings.py` currently uses a random placeholder for `pyramid_correlation`, so experiment comparisons should treat that metric as noisy until the project provides a real scorer.
- The current contrastive setup uses project IDs only within a batch, which is acceptable for in-batch positives but likely weak when batches contain few duplicate projects.
- The extracted `data/corpus.jsonl` does not expose `project_number`; it only has `id`, `text`, and `filename`, so the original contrastive labels were effectively broken for the real corpus.
- This sandbox exposes Windows Store Python aliases, but they are not executable here; `uv` is also unavailable, so runtime validation is currently environment-blocked.
