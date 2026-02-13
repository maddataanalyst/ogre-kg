# OGRE KG

Ontology-Grounded Retrieval & Enrichment for Knowledge Graphs (OGRE KG) is an extension-oriented Python library for graph-aware retrieval and ontology-focused knowledge graph processing on top of LlamaIndex.

The project focuses on:
- Entity disambiguation and controlled entity merging.
- Advanced graph retrievers that use graph context during query-time retrieval.
- Conversion workflows between property graphs and heterogeneous graph representations.

The library is designed for backend portability. Current work targets Memgraph first, with abstractions prepared for Neo4j and local/in-memory property graph workflows.

## Installation (UV)

```bash
uv sync
```

To include development tooling:

```bash
uv sync --group dev
```

To include research/notebook tooling:

```bash
uv sync --group research
```

## Quality Gates

Continuous integration runs `ruff` and `pytest` on every pull request and push to `main`.

Enable local git hooks with pre-commit:

```bash
uv sync --group dev
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

## Import Path

Use the package namespace import path:

```python
from ogre_kg.kg_processors import MemgraphNodePathContextRetriever
```

## Status

This repository is in active early-stage development, with emphasis on robust abstractions, testability, and good coding standards.
