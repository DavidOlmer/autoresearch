# Rebel TOV Autoresearch

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for training on Rebel Group's consulting documents.

## Goals

**Phase A: Embeddings**
- Train sentence embeddings on 690 Rebel final reports
- Cluster by client type × project type
- Enable nearest-neighbor lookup for new projects

**Phase B: TOV (Tone of Voice)**
- Fine-tune LLM on Rebel writing style
- Enforce pyramid writing principles
- Generate text that "sounds like Rebel"

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/DavidOlmer/autoresearch.git
cd autoresearch
uv sync

# 2. Extract corpus from Rebel documents (local machine with access)
uv run scripts/extract_corpus.py

# 3. Run Phase A (embeddings)
uv run train_embeddings.py

# 4. Run Phase B (TOV fine-tuning)
uv run train_tov.py
```

## For Codex/Claude Agents

Read `program.md` for full instructions. Key points:

1. **Multi-agent loop**: Research → Hypothesis → Experiment
2. **Search papers** before each experiment (arXiv, HuggingFace, Semantic Scholar)
3. **Generate 3 hypotheses**, rank, execute top one
4. **5-minute time budget** per experiment
5. **NEVER STOP** until manually interrupted

### Phase A Metrics
- `cluster_silhouette`: Cluster quality (-1 to 1, higher better)
- `pyramid_correlation`: Correlation with pyramid scores

### Phase B Metrics
- `tov_score`: Style consistency (0-1)
- `pyramid_score`: Structure compliance (0-1)
- `perplexity`: Language model quality (lower better)

## Project Structure

```
├── CONTEXT.md           # Full project context for agents
├── program.md           # Agent instructions
├── train_embeddings.py  # Phase A: agent modifies this
├── train_tov.py         # Phase B: agent modifies this
├── prepare.py           # Original Karpathy prep (reference)
├── scripts/
│   └── extract_corpus.py  # Extract text from Word/PDF/PPT
├── data/
│   └── corpus.jsonl     # Extracted documents (generated)
└── .github/
    └── workflows/
        └── autoresearch.yml  # CI/CD pipeline
```

## CI/CD

GitHub Actions workflow:
- Triggers on push to `autoresearch/**` branches
- Runs experiments and uploads logs
- **Creates GitHub Issue on failure** (so you can track and fix)
- Updates tracking issue on success

## Data Source

Documents from `C:\Users\David.Olmer\Rebelgroup\REP - Documents\1. Projecten`:
- 312 Word documents
- 304 PDFs
- 74 PowerPoint presentations
- Pattern: `*final*`, `*definitief*`, `*eindrapport*`

## License

Copyright (c) 2026 Rebel Group. All rights reserved.

Internal use only. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT).
