# Rebel TOV Autoresearch — Agent Instructions

This is an autonomous research system for training models on Rebel Group's consulting documents.

## Mission

Train models that:
1. **Phase A**: Generate document embeddings clustered by client × project type
2. **Phase B**: Generate text in Rebel's tone of voice with pyramid writing compliance

## Setup

Before starting experiments:

1. **Create branch**: `git checkout -b autoresearch/<date>-<phase>` (e.g., `autoresearch/mar16-embeddings`)
2. **Read context**:
   - `CONTEXT.md` — full project context and goals
   - `README.md` — original autoresearch setup
   - `data/corpus.jsonl` — extracted Rebel documents
3. **Verify data**: Check `data/corpus.jsonl` exists and has content
4. **Initialize results**: Create `results.tsv` with header row
5. **Confirm baseline**: Run baseline experiment first

## Multi-Agent Loop

You operate as THREE agents in sequence:

### 1. RESEARCH AGENT

Before each experiment, search for relevant techniques:

**Sources to query:**
- arXiv: `https://arxiv.org/search/?query=<terms>&searchtype=all`
- HuggingFace: `https://huggingface.co/models?search=<terms>`
- Semantic Scholar: `https://api.semanticscholar.org/graph/v1/paper/search?query=<terms>`
- Recent blogs: sentence-transformers, MTEB leaderboard, fine-tuning guides

**Search terms by phase:**
- Phase A: "sentence embeddings clustering", "domain-specific embeddings", "contrastive learning documents", "MTEB benchmark", "document similarity"
- Phase B: "style transfer LLM", "tone of voice fine-tuning", "PEFT LoRA writing style", "constitutional AI writing"

**Output**: List 2-3 relevant papers/techniques with key insights.

### 2. HYPOTHESIS AGENT

Generate hypotheses based on research:

**Template:**
```
HYPOTHESIS: [One sentence description]
RATIONALE: [Why this might work, based on research]
IMPLEMENTATION: [Specific code changes needed]
EXPECTED IMPACT: [High/Medium/Low] on [metric]
RISK: [What could go wrong]
```

**Generate 3 hypotheses, then rank by:**
- Expected impact (1-5)
- Implementation effort (1-5, lower is better)
- Risk level (1-5, lower is better)

**Score = Impact × (6 - Effort) × (6 - Risk)**

Pick the highest scoring hypothesis.

### 3. EXPERIMENT AGENT

Execute the top hypothesis:

1. **Modify code**: Edit `train_embeddings.py` (Phase A) or `train_tov.py` (Phase B)
2. **Commit**: `git commit -am "Experiment: <description>"`
3. **Run**: `uv run train_embeddings.py > run.log 2>&1` (or `train_tov.py`)
4. **Evaluate**: Extract metrics from log
5. **Decide**: Keep (improved) or Discard (same/worse)
6. **Log**: Append to `results.tsv`

## Metrics

### Phase A (Embeddings)
```
cluster_silhouette:  0.35      # Cluster quality (-1 to 1, higher better)
pyramid_correlation: 0.62      # Correlation with pyramid scores
training_seconds:    300.1
peak_vram_mb:        8192
```

### Phase B (TOV Generation)
```
tov_score:           0.78      # Style consistency (0-1)
pyramid_score:       0.65      # Structure compliance (0-1)
perplexity:          12.3      # Lower is better
training_seconds:    300.1
```

## Results Format

Tab-separated `results.tsv`:
```
commit	metric1	metric2	memory_gb	status	hypothesis	research_source
a1b2c3d	0.350	0.620	8.0	keep	baseline	-
b2c3d4e	0.380	0.650	8.2	keep	Add hard negatives from same project	arxiv:2201.xxxxx
c3d4e5f	0.340	0.600	8.0	discard	Switch to MiniLM base	HF:sentence-transformers
```

## Constraints

**What you CAN do:**
- Modify `train_embeddings.py` or `train_tov.py`
- Search papers and HuggingFace for techniques
- Try different model architectures, loss functions, hyperparameters
- Add new training strategies (contrastive, triplet, etc.)

**What you CANNOT do:**
- Modify `prepare.py` or `scripts/extract_corpus.py`
- Install new packages (only what's in `pyproject.toml`)
- Modify the evaluation functions
- Skip the research phase

## NEVER STOP

Once experiments begin:
- Do NOT pause to ask the human
- Do NOT ask "should I continue?"
- If stuck, search for more papers
- If out of ideas, try combining previous near-misses
- Run until manually interrupted

**Target: 12 experiments/hour, 100 experiments overnight**

## Experiment Ideas (Starting Points)

### Phase A — Embeddings
1. Different base models (MiniLM, MPNet, E5, BGE)
2. Contrastive loss with hard negatives (same project = positive, different project = negative)
3. Multi-task: cluster loss + pyramid score prediction
4. Domain adaptation: continue pretraining on Rebel corpus
5. Pooling strategies (mean, CLS, attention-weighted)

### Phase B — TOV
1. LoRA fine-tuning on different layers
2. Style tokens / control codes
3. Constitutional AI: reject non-pyramid outputs
4. Reward model trained on human-rated Rebel texts
5. Mixture of experts: one per document type

## Logging

After each experiment, update `experiment_log.md`:

```markdown
## Experiment <N> — <timestamp>

### Research
- Paper 1: [title] — key insight
- Paper 2: [title] — key insight

### Hypothesis
<selected hypothesis>

### Result
- Metric: X.XXX (baseline: Y.YYY, delta: +/-Z.ZZZ)
- Status: keep/discard
- Learning: <what did we learn>

### Next
- Promising direction: <based on this result>
```

## Quick Reference

```bash
# Run Phase A experiment
uv run train_embeddings.py > run.log 2>&1

# Run Phase B experiment
uv run train_tov.py > run.log 2>&1

# Check metrics
grep "^cluster_silhouette:\|^pyramid_correlation:" run.log

# Keep experiment
git add -A && git commit -m "Keep: <description>"

# Discard experiment
git reset --hard HEAD~1
```
