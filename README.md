# CogitoRAG

> **Anonymous code release accompanying our EMNLP submission. All author and
> affiliation information has been removed for double-blind review.**

This repository contains the reference implementation of **CogitoRAG**, the
retrieval-augmented generation framework described in the accompanying paper.
For the motivation, algorithmic details, hyperparameter settings, and
experimental results, please refer to the paper. This README only describes
the layout of the codebase and how to run it.

---

## Code Organization

CogitoRAG is organized around the `main_cog.py` entry point. The
implementation follows the two-stage pipeline described in the paper:

- **Offline indexing.** Raw passages are segmented and transformed into
  passage-grounded semantic memory units, which are then used to construct a
  multi-dimensional graph over entities, facts, memories, and provenance
  passages.
- **Online retrieval.** Given a query, the system performs query
  decomposition, diffusion over the memory graph, and structure-aware
  reranking, and finally assembles evidence as passage–memory pairs for the
  generator.

The mapping between the modules in the paper and the source files in this
repository is summarized below.

| Module in the paper                | Source file (this repository)                              |
|------------------------------------|------------------------------------------------------------|
| Memory extraction (`<think>` / `<memory>`) | `src/legacy/information_extraction/openie_openai.py` |
| Multi-dimensional graph construction       | `src/legacy/TAG.py`                                  |
| Query Decomposition Module (QDM)            | `src/legacy/TAG.py`                                 |
| Entity Diffusion Module (EDF)               | `src/legacy/TAG.py`                                 |
| CogniRank reranking                          | `src/legacy/TAG.py`                                |
| Evidence assembly (passage–memory pairing)  | `src/legacy/TAG.py`                                 |
| Main runner                                 | `main_cog.py`                                       |

`src/cogitorag` re-exports the public-facing API; `src/legacy` hosts the
internal modules.

---

## Project Layout

```text
CogitoRAG/
├── main_cog.py
├── README.md
├── requirements.txt
├── setup.py
├── scripts/
├── reproduce/
│   └── dataset/
└── src/
    ├── cogitorag/              # public-facing entry point
    └── legacy/                 # internal modules
        ├── TAG.py
        ├── LegacyGraphRAG.py
        ├── embedding_store.py
        ├── rerank.py
        ├── information_extraction/
        ├── llm/
        ├── prompts/
        └── utils/
```

---

## Installation

```bash
pip install -r requirements.txt
```

Configure the LLM and embedding backends through environment variables:

```bash
export OPENAI_API_KEY=your_key
export LLM_NAME=gpt-4o-mini
export EMBEDDING_NAME=nvidia/NV-Embed-v2
```

For Azure OpenAI:

```bash
export AZURE_ENDPOINT=your_azure_chat_endpoint
export AZURE_EMBEDDING_ENDPOINT=your_azure_embedding_endpoint
```

---

## Data

Sample data is kept under `reproduce/dataset`. The default data layout is:

- `reproduce/dataset/{dataset}.json`
- `reproduce/dataset/{dataset}_corpus.json`

---

## Notes on This Release

This release has been cleaned of:

- experiment logs and large output files
- caches and `__pycache__` directories
- one-off experiment scripts and intermediate notes
- local-environment artifacts
