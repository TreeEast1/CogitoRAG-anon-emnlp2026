# CogitoRAG: Retrieval After Comprehension

> **Anonymous code release accompanying our EMNLP submission. All author and affiliation information has been removed for double-blind review.**

This repository provides the reference implementation of **CogitoRAG**, a retrieval-augmented generation framework that addresses a fundamental limitation of conventional RAG systems: retrieving raw text chunks or knowledge triples *before* semantic understanding, leading to fragmented evidence, ambiguity propagation, and localized reasoning. 

Instead of directly indexing raw passages, CogitoRAG follows the principle of **retrieval after comprehension**: it first transforms unstructured corpora into **Semantic Gist Memories** — compact, disambiguated memory units that consolidate explicit facts while resolving implicit relations and contextual ambiguities. These memories are organized into a multi-dimensional graph integrating entities, facts, semantic memories, and provenance passages, enabling globally coherent retrieval over semantically consolidated representations.

---

## Core Idea

Conventional RAG systems typically retrieve raw text chunks or knowledge triples before semantic understanding. This leads to:
- **Fragmented evidence**: Retrieved passages often contain unresolved references and ambiguous mentions
- **Ambiguity propagation**: Implicit relations and contextual dependencies are lost during retrieval
- **Localized reasoning**: Models reason over disconnected textual units rather than coherent semantic scenes

CogitoRAG addresses these limitations by introducing **Semantic Gist Memory** as the retrieval unit. Inspired by cognitive theories of human memory (Reyna and Brainerd, 1995; Tulving et al., 1972; Kintsch and Van Dijk, 1978), our framework consolidates information into semantically meaningful "gist" representations that preserve essential semantic structure while filtering superficial details.

---

## Framework Overview

CogitoRAG follows a **two-stage paradigm**:

### Stage I: Semantic Memory Consolidation
Transforms raw passages into passage-grounded semantic memories that:
- Resolve references and clarify implicit relations
- Preserve provenance to the original passage
- Produce KG-friendly structured representations
- Handle both direct factual passages (light normalization) and complex narratives requiring deeper comprehension (strict disambiguation)

### Stage II: Memory-grounded Retrieval
Given a query, performs:
1. **Query Decomposition Module (QDM)**: Decomposes complex multi-hop queries into sub-questions aligned with memory units
2. **Entity Diffusion Module (EDF)**: Propagates query semantics over the multi-dimensional graph via personalized PageRank with restart
3. **CogniRank**: Structure-aware reranking that jointly models graph topological relevance and semantic similarity
4. **Evidence Assembly**: Pairs retrieved passages with their corresponding gist memories to provide both faithful grounding and high-density semantic support

---

## Key Innovations

| Component | Description | Implementation |
|-----------|-------------|-----------------|
| **Semantic Gist Memory** | Passage-grounded memory units that capture salient information while maintaining provenance | `src/legacy/information_extraction/openie_openai.py` |
| **Multi-dimensional Graph** | Integrates entities (V), memories (M), relations (E), facts (F), and passages (P) | `src/legacy/TAG.py` |
| **Query Decomposition Module (QDM)** | Decomposes complex queries into sub-questions | `src/legacy/TAG.py` (online stage) |
| **Entity Diffusion Module (EDF)** | Spreads query activation over the memory graph via random-walk-with-restart | `src/legacy/TAG.py` |
| **CogniRank** | Balances diffusion-derived structural relevance and direct semantic similarity | `src/legacy/TAG.py` |
| **Evidence Assembly** | Pairs retrieved passages with corresponding memories | `main_cog.py` |

---

## Project Structure

```
CogitoRAG/
├── main_cog.py              # Main entry point
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup (anonymized)
├── reproduce/               # Reproduction scripts and data
│   └── dataset/            # Benchmark datasets (NQ, PopQA, MuSiQue, 2Wiki, HotpotQA)
├── scripts/                 # Utility scripts
└── src/
    ├── cogitorag/          # Public API
    └── legacy/             # Internal implementation
        ├── TAG.py          # Core graph construction and retrieval (EDF, CogniRank)
        ├── LegacyGraphRAG.py
        ├── rerank.py       # Reranking utilities
        ├── embedding_model/ # Embedding models (NV-Embed-v2)
        ├── llm/           # LLM interfaces (GPT-4o-mini, etc.)
        ├── prompts/        # Prompt templates for memory extraction and QA
        └── utils/         # Utilities
```

---

## Installation

```bash
pip install -r requirements.txt
```

Configure LLM and embedding backends via environment variables:

```bash
# OpenAI-compatible API
export OPENAI_API_KEY=your_key
export LLM_NAME=gpt-4o-mini
export EMBEDDING_NAME=nvidia/NV-Embed-v2

# Or use Azure OpenAI
export AZURE_ENDPOINT=your_azure_chat_endpoint
export AZURE_EMBEDDING_ENDPOINT=your_azure_embedding_endpoint
export AZURE_OPENAI_API_KEY=your_key
```

---

## Data Format

Sample data is provided under `reproduce/dataset`. The expected format:

- `{dataset}.json`: Query file with `id`, `question`, `answer` (for evaluation)
- `{dataset}_corpus.json`: Corpus file with `title` and `text` fields

**Supported benchmarks**:
- **Simple QA**: Natural Questions (NQ), PopQA
- **Multi-hop QA**: MuSiQue, 2WikiMultiHopQA, HotpotQA
- **GraphBench**: Novel, Medical (evaluates broader multi-task generation beyond QA)

Dataset statistics (1,000 queries uniformly sampled per benchmark):

| Dataset | Queries | Passages |
|----------|---------|-----------|
| NQ | 1,000 | 9,633 |
| PopQA | 1,000 | 8,676 |
| MuSiQue | 1,000 | 11,656 |
| 2WikiMultiHopQA | 1,000 | 6,119 |
| HotpotQA | 1,000 | 9,811 |
| GraphBench Novel | 2,010 | 1,108 |
| GraphBench Medical | 2,062 | 224 |

---

## Usage

### Quick Start

```bash
# Run CogitoRAG on a benchmark dataset
python main_cog.py \
  --dataset musique \
  --llm gpt-4o-mini \
  --embedding nvidia/NV-Embed-v2 \
  --top_k 5
```

### Key Parameters

- `--dataset`: Dataset name (nq, popqa, musique, 2wiki, hotpotqa)
- `--llm`: LLM backend (gpt-4o-mini, etc.)
- `--embedding`: Embedding model (NV-Embed-v2)
- `--top_k`: Number of passages to retrieve (default: 5)
- `--alpha`: Frequency reward magnitude in entity activation (default: 2.0)
- `--beta`: Reward saturation rate (default: 1.0)
- `--gamma`: Restart probability for diffusion (default: 0.1)
- `--epsilon`: Fusion coefficient balancing diffusion and semantic similarity (default: 0.95)

---

## Experimental Results

CogitoRAG is evaluated against **9 baseline methods** on 5 QA benchmarks and GraphBench:

### Baselines
1. **None**: No retrieval, relying solely on parametric knowledge
2. **NV-Embed-v2** (Lee et al., 2025): Dense retrieval baseline
3. **GraphRAG** (Edge et al., 2024): Graph-based RAG
4. **LightRAG** (Guo et al., 2024): Lightweight graph RAG
5. **RAPTOR** (Sarthi et al., 2024): Recursive abstractive processing
6. **HippoRAG** (Wang et al., 2024a): Biomimetic graph RAG
7. **HippoRAG2** (Gutiérrez et al., 2025): Improved HippoRAG
8. **ComoRAG** (Wang et al., 2025b): Community-based memory RAG
9. **ToG2** (Ma et al., 2025): Think-on-Graph 2.0

### Main Results (EM / F1)

| Method | NQ | PopQA | MuSiQue | 2Wiki | HotpotQA |
|--------|----|-------|----------|-------|-----------|
| None | 35.20 / 52.70 | 16.10 / 22.70 | 11.20 / 22.00 | 30.20 / 36.30 | 28.60 / 41.00 |
| NV-Embed-v2 | 43.50 / 59.90 | 41.70 / 55.80 | 32.80 / 46.00 | 54.40 / 60.80 | 57.30 / 71.00 |
| GraphRAG | 38.00 / 55.50 | 30.70 / 51.30 | 27.00 / 42.00 | 45.70 / 61.00 | 51.40 / 67.60 |
| HippoRAG2 | 43.40 / 60.00 | 41.70 / 55.70 | 35.00 / 49.30 | 60.50 / 69.70 | 56.30 / 71.10 |
| **CogitoRAG (Ours)** | **51.30 / 63.56** | **50.94 / 59.94** | **43.20 / 53.95** | **69.90 / 76.20** | **60.70 / 73.29** |

**Key findings**:
- CogitoRAG consistently outperforms existing RAG baselines across all datasets
- Largest improvements on multi-hop datasets (MuSiQue: +8.2 EM, 2Wiki: +9.4 EM)
- GraphBench results show superior performance on fact retrieval (FR), creative generation (CR), and case study (CS) tasks

### Ablation Study (MuSiQue)

| Setting | Traditional KG | Summary | Ours |
|---------|-----------------|---------|------|
| w/o EDF | 31.90 / 46.46 | 29.10 / 43.80 | 35.30 / 49.60 |
| w/o CogniRank | 32.40 / 47.02 | 29.20 / 43.26 | 36.50 / 50.23 |
| w/o QDM | 31.70 / 46.27 | 29.90 / 44.62 | 41.70 / 53.01 |
| **Full** | **35.00 / 49.30** | **36.50 / 50.20** | **43.20 / 53.95** |

Ablation results confirm that:
1. Semantic Gist Memory provides a stronger retrieval foundation than Traditional KG or Summary-based graphs
2. EDF, CogniRank, and QDM address complementary bottlenecks in multi-hop retrieval
3. The joint effect of memory-based graph construction and retrieval-time evidence reasoning yields the best performance

---

## Hyperparameter Sensitivity

Experiments on 2WikiMultiHopQA show that CogitoRAG is robust to hyperparameter choices:

- **α (frequency reward magnitude)**: Optimal at α = 2.0; moderate guidance without narrow calibration
- **β (reward saturation rate)**: Optimal at β = 1.0
- **ε (fusion coefficient)**: Diffusion-only (ε = 1.0) already achieves strong results; introducing small semantic component (ε = 0.95) improves performance

---

## Reproducibility Notes

This release has been prepared for anonymous review:

- Experimental logs and raw outputs have been removed
- Hyperparameter configurations are omitted (see paper for details)
- Cached intermediate results and `__pycache__` directories are excluded
- All author and institution identifiers have been anonymized
- The `setup.py` file has been modified to remove author information

For reproducibility, we provide the core implementation and sample data. Full experimental results and ablation studies are reported in the accompanying paper.

---

## Citation

If you use this code, please refer to our paper (link will be updated upon publication).

```bibtex
% Will be updated after acceptance
```

---

## Ethical Considerations

- This work uses publicly available datasets and does not involve human subjects or newly collected personally identifiable information
- We follow the licenses and terms of use of the datasets and APIs used in our experiments
- Since CogitoRAG relies on LLM-based memory extraction and generation, its outputs may inherit biases, factual errors, or unsupported associations from the underlying models
- The proposed framework should not be directly deployed in high-stakes applications without appropriate human oversight

---

## License

This code is released under the MIT License for academic use.

---

## Anonymous Repository

Code and data are anonymously available at:  
https://anonymous.4open.science/r/CogitoRAG-anon-emnlp2026-5543/

(Anonymized for double-blind review)
