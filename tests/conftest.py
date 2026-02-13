"""Shared test fixtures for OGRE KG tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class FakeStructuredStore:
    """Fake graph store supporting structured queries for testing.

    Backend-agnostic — use ``FakeMemgraphStore`` or ``FakeNeo4jStore`` when
    backend detection is required.
    """

    supports_structured_queries: bool = True
    similarity_pairs: list[dict] = field(default_factory=list)
    merge_result: list[dict] = field(default_factory=lambda: [{"node": {"name": "merged"}}])

    def __post_init__(self) -> None:
        self.queries: list[str] = []

    def structured_query(self, query: str) -> list[dict]:
        self.queries.append(query)
        if "node_similarity.cosine" in query or "vector.similarity" in query:
            return self.similarity_pairs
        if "refactor.merge_nodes" in query or "apoc.refactor.mergeNodes" in query:
            return self.merge_result
        return []

    async def astructured_query(self, query: str) -> list[dict]:
        return self.structured_query(query)


@dataclass
class FakeMemgraphStore(FakeStructuredStore):
    """Fake Memgraph store — detected as MEMGRAPH by ``detect_graph_store_backend``."""


@dataclass
class FakeNeo4jStore(FakeStructuredStore):
    """Fake Neo4j store — detected as NEO4J by ``detect_graph_store_backend``."""


@dataclass
class FakeIndex:
    """Fake index-like object wrapping a property graph store."""

    property_graph_store: FakeStructuredStore


@pytest.fixture
def fake_store() -> FakeStructuredStore:
    return FakeStructuredStore()


@pytest.fixture
def fake_memgraph_store() -> FakeMemgraphStore:
    return FakeMemgraphStore()


@pytest.fixture
def fake_neo4j_store() -> FakeNeo4jStore:
    return FakeNeo4jStore()


@pytest.fixture
def fake_store_with_pairs() -> FakeStructuredStore:
    return FakeStructuredStore(
        similarity_pairs=[
            {"name1": "IBM", "name2": "International Business Machines", "similarity": 0.9},
            {"name1": "AI", "name2": "Artificial Intelligence", "similarity": 0.8},
        ]
    )


@pytest.fixture
def fake_index(fake_store: FakeStructuredStore) -> FakeIndex:
    return FakeIndex(property_graph_store=fake_store)
