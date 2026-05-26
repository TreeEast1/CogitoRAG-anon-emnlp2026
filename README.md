# CogitoRAG

> **Anonymous code release accompanying our EMNLP submission. All author and
> affiliation information has been removed for double-blind review.**

CogitoRAG is a graph-enhanced retrieval-augmented question answering
system. The repository is organized around the `main_cog.py` entry point and
keeps three core design choices that together define the method:

1. A **Memory + Think** dual-output mechanism at the indexing stage.
2. **Entity-frequency-aware** scoring at the retrieval stage.
3. **Gamma-fusion** reranking that linearly combines graph-based and dense
   retrieval scores.

The main entry point is `main_cog.py`, and the core implementation lives in
`src/legacy/TAG.py`. The package is exposed as `cogitorag`, with
`src/cogitorag` as the public-facing module and `src/legacy` as an internal
compatibility layer that wraps the underlying graph-based RAG framework on
which CogitoRAG is built (cited in the accompanying paper).

---

## Core Components of CogitoRAG

### 1. Indexing — Memory + Think

During OpenIE, each document is decomposed into:

- `think`: the model's reasoning trace, kept for inspection only and **not**
  used for graph construction.
- `memory`: a structured summary that is consumed by the downstream graph
  builder.

Relevant files:

- `main_cog.py`
- `src/legacy/TAG.py`
- `src/legacy/information_extraction/openie_openai.py`

### 2. Retrieval — entity-frequency enhancement

At retrieval time, scores are augmented with entity-frequency statistics.
This is one of the two retrieval-side innovations in CogitoRAG.

Relevant file: `src/legacy/TAG.py`.

### 3. Reranking — gamma fusion

The final ranking blends the graph-based PPR score and the dense retrieval
score via a tunable mixing weight.

Relevant file: `src/legacy/TAG.py`.

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
    └── legacy/                 # internal compatibility layer
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

It keeps:

- the `main_cog.py` main entry of CogitoRAG
- the `TAG.py` core logic
- the key prompts
- the sample datasets and basic utility scripts
