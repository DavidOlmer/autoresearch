# Experiment Log

## Experiment 0 - 2026-03-16

### Research
- Paper 1: Text Embeddings by Weakly-Supervised Contrastive Pre-training - weak supervision at scale can produce strong retrieval and clustering embeddings without hand-labeled pairs.
- Paper 2: BGE M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity - multilingual embedding models are useful when the corpus mixes Dutch and English and when retrieval granularity varies.
- Paper 3: MTEB: Massive Text Embedding Benchmark - model choice matters materially, and stronger embedding backbones often outperform MiniLM on clustering-style benchmarks.

### Hypotheses
HYPOTHESIS 1: Infer pseudo-labels from `filename` and train contrastively on document families instead of the nonexistent `project_number`.
RATIONALE: The corpus lacks explicit project metadata, so the current loss is training on bad labels; weakly supervised contrastive learning suggests that noisy positives can still be useful when they are systematic.
IMPLEMENTATION: Add deterministic filename normalization, use the resulting document-family label in the dataset, and preserve high-signal headings in the text preview.
EXPECTED IMPACT: High on cluster_silhouette.
RISK: Filename normalization may merge unrelated documents or split true document families.
SCORE: 80 (impact 5, effort 2, risk 2)

HYPOTHESIS 2: Switch from `all-MiniLM-L6-v2` to a multilingual embedding backbone such as `multilingual-e5-base` or `bge-m3`.
RATIONALE: The corpus is primarily Dutch with some English, and recent embedding models benchmark better than MiniLM on multilingual tasks.
IMPLEMENTATION: Change the base model, update max sequence length if needed, and retune batch size.
EXPECTED IMPACT: Medium to high on cluster_silhouette.
RISK: More VRAM use and longer iteration time within the five-minute budget.
SCORE: 48 (impact 4, effort 3, risk 3)

HYPOTHESIS 3: Replace raw truncation with structure-aware previews that retain section headings, summary sections, and conclusions before tokenization.
RATIONALE: Rebel reports are long and highly structured, so a representative preview should preserve top-down sections instead of taking only the first flat character span.
IMPLEMENTATION: Build a preview function that prioritizes headings and summary sections before truncation.
EXPECTED IMPACT: Medium on cluster_silhouette.
RISK: Headings may dominate the preview and drown out the body text.
SCORE: 45 (impact 3, effort 2, risk 1)

### Selected
Hypothesis 1 ranked highest and was implemented together with the lower-risk part of Hypothesis 3 because both changes operate on the same dataset preprocessing path.

### Result
- Status: blocked
- Learning: the baseline could not be executed because `uv` is not installed in this environment and the available Windows Store Python aliases are not runnable inside the sandbox.
- Learning: the current corpus schema confirms that `project_number` is absent, so the original contrastive supervision path was invalid for real Phase A data.

### Next
- Run `python -m pytest tests/test_train_embeddings.py` once a working Python interpreter is available.
- Run `uv run train_embeddings.py` or an equivalent Python command to collect the first comparable metrics.
- If the weak-supervision labels help, test a multilingual model backbone next.
