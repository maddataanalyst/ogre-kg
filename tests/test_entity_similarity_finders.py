"""Tests for ogre_kg.kg_processors.entity_similarity_finders module."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from conftest import FakeMemgraphStore, FakeNeo4jStore, FakeStructuredStore
from llama_index.core.graph_stores.types import EntityNode

from ogre_kg.kg_processors.entity_similarity_finders import (
    FuzzyEntitySimilarityFinder,
    MemgraphCypherEntitySimilarityFinder,
    Neo4jGDSEntitySimilarityFinder,
)


@dataclass
class FakeGraphStoreWithGet:
    """Fake store that supports get() returning EntityNode objects."""

    entities: list[EntityNode] = field(default_factory=list)
    supports_structured_queries: bool = True

    def get(self) -> list[EntityNode]:
        return self.entities

    def structured_query(self, query: str) -> list[dict]:
        return []


class TestMemgraphCypherEntitySimilarityFinder:
    def test_find_similar_entities_uses_cosine_query(self):
        # given
        store = FakeMemgraphStore(
            similarity_pairs=[
                {"name1": "IBM", "name2": "International Business Machines", "similarity": 0.9},
            ]
        )
        finder = MemgraphCypherEntitySimilarityFinder(
            source=store, embedding_attr="embedding", similarity_threshold=0.7
        )

        # when
        groups = finder.find_similar_entities()

        # then
        assert len(groups) == 1
        assert {"IBM", "International Business Machines"} in groups
        assert "node_similarity.cosine" in store.queries[0]

    def test_connected_components_via_union_find(self):
        # given - A-B and B-C should merge into one group
        store = FakeMemgraphStore(
            similarity_pairs=[
                {"name1": "A", "name2": "B", "similarity": 0.9},
                {"name1": "B", "name2": "C", "similarity": 0.85},
            ]
        )
        finder = MemgraphCypherEntitySimilarityFinder(
            source=store, embedding_attr="embedding", similarity_threshold=0.5
        )

        # when
        groups = finder.find_similar_entities()

        # then
        assert len(groups) == 1
        assert {"A", "B", "C"} in groups

    def test_threshold_embedded_in_query(self):
        # given
        store = FakeMemgraphStore()
        finder = MemgraphCypherEntitySimilarityFinder(
            source=store, embedding_attr="emb", similarity_threshold=0.42
        )

        # when
        finder.find_similar_entities()

        # then
        assert "0.42" in store.queries[0]

    def test_empty_pairs_returns_empty(self):
        # given
        store = FakeMemgraphStore(similarity_pairs=[])
        finder = MemgraphCypherEntitySimilarityFinder(source=store, embedding_attr="embedding")

        # when
        groups = finder.find_similar_entities()

        # then
        assert groups == []

    def test_rejects_neo4j_store(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Memgraph graph store"):
            MemgraphCypherEntitySimilarityFinder(source=store, embedding_attr="embedding")

    def test_rejects_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            MemgraphCypherEntitySimilarityFinder(source=store, embedding_attr="embedding")


class TestNeo4jGDSEntitySimilarityFinder:
    def test_find_similar_entities_uses_vector_similarity_query(self):
        # given
        store = FakeNeo4jStore(
            similarity_pairs=[
                {"name1": "IBM", "name2": "International Business Machines", "similarity": 0.9},
            ]
        )
        finder = Neo4jGDSEntitySimilarityFinder(
            source=store, embedding_attr="embedding", similarity_threshold=0.5
        )

        # when
        groups = finder.find_similar_entities()

        # then
        assert len(groups) == 1
        assert {"IBM", "International Business Machines"} in groups
        assert "vector.similarity.cosine" in store.queries[0]

    def test_threshold_and_metric_embedded_in_query(self):
        # given
        store = FakeNeo4jStore()
        finder = Neo4jGDSEntitySimilarityFinder(
            source=store,
            embedding_attr="emb",
            similarity_threshold=0.7,
            similarity_metric="euclidean",
        )

        # when
        finder.find_similar_entities()

        # then
        query = store.queries[0]
        assert "0.7" in query
        assert "vector.similarity.euclidean" in query

    def test_connected_components_via_union_find(self):
        # given
        store = FakeNeo4jStore(
            similarity_pairs=[
                {"name1": "X", "name2": "Y", "similarity": 0.9},
                {"name1": "Y", "name2": "Z", "similarity": 0.85},
            ]
        )
        finder = Neo4jGDSEntitySimilarityFinder(
            source=store, embedding_attr="embedding", similarity_threshold=0.5
        )

        # when
        groups = finder.find_similar_entities()

        # then
        assert len(groups) == 1
        assert {"X", "Y", "Z"} in groups

    def test_rejects_memgraph_store(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Neo4j graph store"):
            Neo4jGDSEntitySimilarityFinder(source=store, embedding_attr="emb")

    def test_rejects_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            Neo4jGDSEntitySimilarityFinder(source=store, embedding_attr="emb")


class TestFuzzyEntitySimilarityFinder:
    def test_fuzzy_matching_groups_similar_names(self):
        # given
        store = FakeGraphStoreWithGet(
            entities=[
                EntityNode(name="New York"),
                EntityNode(name="New York City"),
                EntityNode(name="London"),
            ]
        )
        finder = FuzzyEntitySimilarityFinder(source=store, similarity_threshold=70.0)

        # when
        groups = finder.find_similar_entities()

        # then
        assert len(groups) == 1
        assert "New York" in groups[0]
        assert "New York City" in groups[0]

    def test_high_threshold_finds_nothing(self):
        # given
        store = FakeGraphStoreWithGet(
            entities=[
                EntityNode(name="Apple"),
                EntityNode(name="Banana"),
                EntityNode(name="Cherry"),
            ]
        )
        finder = FuzzyEntitySimilarityFinder(source=store, similarity_threshold=99.0)

        # when
        groups = finder.find_similar_entities()

        # then
        assert groups == []

    def test_empty_store_returns_empty(self):
        # given
        store = FakeGraphStoreWithGet(entities=[])
        finder = FuzzyEntitySimilarityFinder(source=store)

        # when
        groups = finder.find_similar_entities()

        # then
        assert groups == []
