from __future__ import annotations

from dataclasses import dataclass

import pytest
from conftest import FakeFalkorDBStore, FakeMemgraphStore, FakeNeo4jStore, FakeStructuredStore

from ogre_kg.kg_processors.graph_db_utils import (
    FalkorDBFulltextIndexBuilder,
    GraphTextIndexSpec,
    MemgraphTextIndexBuilder,
    Neo4jFulltextIndexBuilder,
    default_retriever_index_specs,
    make_graph_text_index_builder,
)


@dataclass
class RaisingMemgraphStore(FakeMemgraphStore):
    def structured_query(self, query: str) -> list[dict]:
        self.queries.append(query)
        raise RuntimeError("index already exists")


@dataclass
class AnalyticalModeMemgraphStore(FakeMemgraphStore):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._raised = False

    def structured_query(self, query: str) -> list[dict]:
        self.queries.append(query)
        if query.startswith("CREATE TEXT INDEX entity_name") and not self._raised:
            self._raised = True
            raise RuntimeError("Text index is not supported in analytical storage mode.")
        return []


class FakeResult:
    def __iter__(self):
        return iter(())


class FakeSession:
    def __init__(self, store: ClientMemgraphStore) -> None:
        self._store = store

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def run(self, query: str) -> FakeResult:
        self._store.session_queries.append(query)
        return FakeResult()


class FakeClient:
    def __init__(self, store: ClientMemgraphStore) -> None:
        self._store = store

    def session(self, database=None) -> FakeSession:
        return FakeSession(self._store)


@dataclass
class ClientMemgraphStore(FakeMemgraphStore):
    _database: str = "memgraph"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.session_queries: list[str] = []
        self.client = FakeClient(self)

    def structured_query(self, query: str) -> list[dict]:
        self.queries.append(query)
        return []


def test_default_retriever_index_specs_contains_chunk_and_entity() -> None:
    specs = default_retriever_index_specs()
    assert specs == [
        GraphTextIndexSpec(name="entity_name", label="__Entity__", properties=("name",)),
        GraphTextIndexSpec(name="chunk_text", label="Chunk", properties=("text",)),
    ]


def test_memgraph_builder_creates_text_indices_with_expected_syntax() -> None:
    store = FakeMemgraphStore()
    builder = MemgraphTextIndexBuilder(source=store)

    builder.create_indices()

    assert store.queries == [
        "CREATE TEXT INDEX entity_name ON :__Entity__(name);",
        "CREATE TEXT INDEX chunk_text ON :Chunk(text);",
    ]


def test_memgraph_builder_handles_existing_index_when_if_not_exists_enabled() -> None:
    store = RaisingMemgraphStore()
    builder = MemgraphTextIndexBuilder(source=store)

    builder.create_indices(if_not_exists=True)

    assert len(store.queries) == 2


def test_memgraph_builder_raises_on_existing_index_when_if_not_exists_disabled() -> None:
    store = RaisingMemgraphStore()
    builder = MemgraphTextIndexBuilder(source=store)

    with pytest.raises(RuntimeError, match="already exists"):
        builder.create_indices(if_not_exists=False)


def test_memgraph_builder_requires_single_property_per_text_index() -> None:
    store = FakeMemgraphStore()
    builder = MemgraphTextIndexBuilder(source=store)
    spec = GraphTextIndexSpec(name="bad", label="Chunk", properties=("text", "title"))

    with pytest.raises(ValueError, match="expects exactly one property"):
        builder.build_create_query(spec)


def test_memgraph_builder_auto_switches_storage_mode_on_analytical_error() -> None:
    store = AnalyticalModeMemgraphStore()
    builder = MemgraphTextIndexBuilder(source=store)

    builder.create_indices()

    assert store.queries == [
        "CREATE TEXT INDEX entity_name ON :__Entity__(name);",
        "STORAGE MODE IN_MEMORY_TRANSACTIONAL;",
        "CREATE TEXT INDEX entity_name ON :__Entity__(name);",
        "CREATE TEXT INDEX chunk_text ON :Chunk(text);",
    ]


def test_memgraph_builder_prefers_autocommit_session_run_when_available() -> None:
    store = ClientMemgraphStore()
    builder = MemgraphTextIndexBuilder(source=store)

    builder.create_indices()

    assert store.session_queries == [
        "CREATE TEXT INDEX entity_name ON :__Entity__(name);",
        "CREATE TEXT INDEX chunk_text ON :Chunk(text);",
    ]
    assert store.queries == []


def test_neo4j_builder_creates_fulltext_indices_with_expected_syntax() -> None:
    store = FakeNeo4jStore()
    builder = Neo4jFulltextIndexBuilder(source=store)

    builder.create_indices()

    assert store.queries == [
        "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.name]",
        "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
    ]


def test_neo4j_builder_drop_query_uses_if_exists() -> None:
    store = FakeNeo4jStore()
    builder = Neo4jFulltextIndexBuilder(source=store)

    builder.drop_indices(index_names=["entity_name", "chunk_text"], if_exists=True)

    assert store.queries == [
        "DROP INDEX entity_name IF EXISTS",
        "DROP INDEX chunk_text IF EXISTS",
    ]


def test_falkordb_builder_creates_fulltext_indices_with_expected_syntax() -> None:
    store = FakeFalkorDBStore()
    builder = FalkorDBFulltextIndexBuilder(source=store)

    builder.create_indices()

    assert store.queries == [
        "CALL db.idx.fulltext.createNodeIndex('__Entity__', 'name')",
        "CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')",
    ]


def test_falkordb_builder_drop_query_uses_expected_syntax() -> None:
    store = FakeFalkorDBStore()
    builder = FalkorDBFulltextIndexBuilder(source=store)

    builder.drop_indices(index_names=["__Entity__", "Chunk"], if_exists=True)

    assert store.queries == [
        "CALL db.idx.fulltext.drop('__Entity__')",
        "CALL db.idx.fulltext.drop('Chunk')",
    ]


def test_make_graph_text_index_builder_resolves_memgraph() -> None:
    builder = make_graph_text_index_builder(FakeMemgraphStore())
    assert isinstance(builder, MemgraphTextIndexBuilder)


def test_make_graph_text_index_builder_resolves_falkordb() -> None:
    builder = make_graph_text_index_builder(FakeFalkorDBStore())
    assert isinstance(builder, FalkorDBFulltextIndexBuilder)


def test_make_graph_text_index_builder_resolves_neo4j() -> None:
    builder = make_graph_text_index_builder(FakeNeo4jStore())
    assert isinstance(builder, Neo4jFulltextIndexBuilder)


def test_make_graph_text_index_builder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Cannot detect backend"):
        make_graph_text_index_builder(FakeStructuredStore())
