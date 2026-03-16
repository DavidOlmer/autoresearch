# Rebel TOV Autoresearch — Context voor Codex

## Missie

Train een model dat:
1. **Fase A**: Embeddings genereert per document, geclusterd op klant × opdrachttype
2. **Fase B**: Tekst genereert in Rebel's tone of voice, conform pyramid writing principles

## Dataset

- **Locatie**: `C:\Users\David.Olmer\Rebelgroup\REP - Documents\1. Projecten`
- **Omvang**: 690 finale documenten (312 .docx, 304 .pdf, 74 .pptx)
- **Selectie**: Bestanden met `final|definitief|eindrapport|rapport.*v\d|deliverable` in naam
- **Structuur**: Projectmappen met format `P[nummer] - [naam]`

## Doelen

### Fase A: Embedding Space
- Train sentence embeddings op Rebel rapporten
- Cluster op klanttype en opdrachttype
- Output: vector per document, nearest neighbor lookup voor nieuwe opdrachten
- **Metric**: `cluster_quality` (silhouette score) + `pyramid_score`

### Fase B: TOV Generation
- Fine-tune LLM op Rebel corpus
- Genereer tekst in Rebel stijl
- **Metric**: `tov_consistency` + `pyramid_compliance`

## Kwaliteitscriteria

### Pyramid Writing Principles
1. **Top-down**: Conclusie eerst, dan onderbouwing
2. **MECE**: Mutually Exclusive, Collectively Exhaustive
3. **SCQ**: Situation, Complication, Question structuur
4. **Grouping**: Max 5 items per niveau
5. **Governing thought**: Elke sectie heeft één kernboodschap

### Rebel Tone of Voice
- Zie `data/tov_guidelines.md` (te extraheren uit bestaande rapporten)
- Zakelijk maar toegankelijk
- Evidence-based, cijfermatig onderbouwd
- Actieve schrijfstijl

## Multi-Agent Architectuur

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│  - Coördineert agents                                        │
│  - Beheert experiment queue                                  │
│  - Aggregeert resultaten                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│RESEARCH│  │HYPOTHESIS│  │EXPERIMENT│
│ AGENT  │  │  AGENT   │  │  AGENT   │
├────────┤  ├──────────┤  ├──────────┤
│- Zoek  │  │- Genereer│  │- Voer    │
│ papers │  │  ideeën  │  │  uit     │
│- Arxiv │  │- Combine │  │- Meet    │
│- HF    │  │  near-   │  │- Log     │
│- Blogs │  │  misses  │  │- Keep/   │
└────────┘  └──────────┘  │  Discard │
                          └──────────┘
```

## Experiment Loop

```
LOOP FOREVER:
1. RESEARCH: Zoek relevante papers/technieken (arxiv, HuggingFace, blogs)
2. HYPOTHESIS: Genereer 3 hypotheses gebaseerd op research
3. RANK: Score hypotheses op verwachte impact × haalbaarheid
4. EXPERIMENT: Voer top hypothesis uit
5. EVALUATE: Meet metrics, vergelijk met baseline
6. DECIDE: Keep (beter) of Discard (slechter/gelijk)
7. LOG: Schrijf naar results.tsv
8. REPEAT
```

## Technische Stack

| Component | Tool |
|-----------|------|
| Text extraction | `python-docx`, `PyMuPDF`, `python-pptx` |
| Embeddings | `sentence-transformers` |
| LLM base | `transformers`, `peft` |
| Metrics | Custom pyramid scorer |
| Search | `arxiv` API, `semanticscholar` |

## Constraints

- **Time budget**: 5 minuten per experiment (conform Karpathy)
- **Hardware**: Single GPU (RTX of beter)
- **Packages**: Alleen wat in `pyproject.toml` staat
- **Files**: Agent wijzigt alleen `train.py` en `train_embeddings.py`

## Success Criteria

### Fase A
- Silhouette score > 0.3 voor klant clusters
- Pyramid score correlatie > 0.6 met human ratings

### Fase B
- BLEU/ROUGE tegen held-out Rebel docs
- Human eval: 4/5 gemiddeld op "klinkt als Rebel"
