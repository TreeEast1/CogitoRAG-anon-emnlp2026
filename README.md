# CogitoRAG

> **Anonymous code release accompanying our EMNLP submission. All author and affiliation information has been removed for double-blind review.**

This repository provides the reference implementation of **CogitoRAG**, a retrieval-augmented generation framework that enhances multi-hop question answering through cognitive-inspired memory indexing and structure-aware retrieval. The accompanying paper describes the motivation, methodological details, and experimental evaluation. This README focuses on codebase organization and reproducibility.

---

## Overview

CogitoRAG addresses the limitation of conventional RAG systems in handling complex multi-hop queries that require iterative reasoning over dispersed evidence. The framework introduces three key innovations:

1. **Semantic Gist Memory**: Passage-grounded memory units that capture salient information while maintaining provenance
2. **Query Decomposition Module (QDM)**: Decomposes complex queries into sub-queries aligned with memory units
3. **Entity Diffusion Module (EDF)**: Propagates query semantics over a multi-dimensional graph via personalized PageRank
4. **CogniRank**: Structure-aware reranking that balances semantic relevance and graph centrality

The implementation follows a two-stage pipeline:

- **Offline Indexing**: Transforms raw passages into semantic memory units and constructs a multi-dimensional graph over entities, facts, memories, and provenance passages.
- **Online Retrieval**: Given a query, performs query decomposition, diffusion-based retrieval over the memory graph, and evidence assembly for the generator.

---

## Module Mapping

The following table maps components described in the paper to their implementations in this repository.

| Component (Paper) | Implementation (This Repo) |
|-------------------|----------------------------|
| Semantic Gist Memory extraction | `src/legacy/information_extraction/openie_openai.py` |
| Multi-dimensional graph construction | `src/legacy/TAG.py` |
| Query Decomposition Module (QDM) | `src/legacy/TAG.py` |
| Entity Diffusion Module (EDF) | `src/legacy/TAG.py` |
| CogniRank reranking | `src/legacy/TAG.py` |
| Evidence assembly | `src/legacy/TAG.py` |
| Main execution pipeline | `main_cog.py` |

`src/cogitorag` provides the public API; `src/legacy` contains internal modules.

---

## Project Structure

```
CogitoRAG/
├── main_cog.py              # Entry point
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── reproduce/               # Reproduction scripts and data
│   └── dataset/            # Benchmark datasets
├── scripts/                 # Utility scripts
└── src/
    ├── cogitorag/          # Public API
    └── legacy/             # Internal implementation
        ├── TAG.py          # Core graph construction and retrieval
        ├── LegacyGraphRAG.py
        ├── rerank.py       # Reranking utilities
        ├── embedding_model/ # Embedding models
        ├── llm/           # LLM interfaces
        ├── prompts/        # Prompt templates
        └── utils/         # Utilities
```

---

## Installation

```bash
pip install -r requirements.txt
```

Configure LLM and embedding backends via environment variables:

```bash
export OPENAI_API_KEY=your_key
export LLM_NAME=gpt-4o-mini
export EMBEDDING_NAME=nvidia/NV-Embed-v2
```

For Azure OpenAI:

```bash
export AZURE_ENDPOINT=your_azure_chat_endpoint
export AZURE_EMBEDDING_ENDPOINT=your_azure_embedding_endpoint
export AZURE_OPENAI_API_KEY=your_key
```

---

## Data Format

Sample data is provided under `reproduce/dataset`. The expected format:

- `{dataset}.json`: Query file with `id`, `question`, `contexts` (for evaluation)
- `{dataset}_corpus.json`: Corpus file with `title` and `text` fields

Supported benchmarks: MuSiQue, HotpotQA, 2WikiMultiHopQA, PopQA.

---

## Reproducibility Notes

This release has been prepared for anonymous review:

- Experimental logs and raw outputs have been removed
- Hyperparameter configurations are omitted (see paper for details)
- Cached intermediate results and `__pycache__` directories are excluded
- All author and institution identifiers have been anonymized

For reproducibility, we provide the core implementation and sample data. Full experimental results and ablation studies are reported in the accompanying paper.

---

## Citation

If you use this code, please refer to our paper (link will be updated upon publication).

---

## License

This code is released under the MIT License for academic use.
