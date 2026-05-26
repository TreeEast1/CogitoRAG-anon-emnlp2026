import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cogitorag",
    version="2.0.0-alpha.2",
    author="Anonymous",
    description="CogitoRAG: a graph-enhanced RAG system with memory-think indexing, entity-frequency-aware retrieval, and gamma fusion reranking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.5.1",
        "transformers==4.45.2",
        "vllm==0.6.6.post1",
        "openai==1.58.1",
        "gritlm==1.0.2",
        "networkx==3.4.2",
        "python_igraph==0.11.8",
        "tiktoken==0.7.0",
        "pydantic==2.10.4",
        "tenacity==8.5.0",
        "einops",  # No version specified
        "tqdm",  # No version specified
    ]
)
