"""Tests for ogre_kg.kg_processors.entity_disambiguation module."""

from __future__ import annotations

import pytest
from conftest import FakeIndex, FakeMemgraphStore, FakeNeo4jStore, FakeStructuredStore

from ogre_kg.kg_processors.entity_disambiguation import EntityDisambiguationProcessor
from ogre_kg.kg_processors.entity_merger import MemgraphEntityMerger, Neo4jEntityMerger
from ogre_kg.kg_processors.entity_similarity_finders import (
    MemgraphCypherEntitySimilarityFinder,
    Neo4jGDSEntitySimilarityFinder,
)
from ogre_kg.utils import resolve_graph_store


class TestResolveGraphStore:
    def test_accepts_direct_store(self):
        # given
        store = FakeStructuredStore()

        # when
        resolved = resolve_graph_store(store)

        # then
        assert resolved is store

    def test_accepts_index_like_objects(self):
        # given
        store = FakeStructuredStore()
        index = FakeIndex(property_graph_store=store)

        # when
        resolved = resolve_graph_store(index)

        # then
        assert resolved is store

    def test_raises_for_invalid_source(self):
        # when / then
        with pytest.raises(ValueError):
            resolve_graph_store("not a store")


class TestEntityDisambiguationProcessorComposition:
    def test_memgraph_finder_with_memgraph_merger(self):
        # given
        store = FakeMemgraphStore(
            similarity_pairs=[
                {"name1": "A", "name2": "B", "similarity": 0.9},
            ]
        )
        finder = MemgraphCypherEntitySimilarityFinder(source=store, embedding_attr="embedding")
        merger = MemgraphEntityMerger(source=store, preview_changes=False)
        processor = EntityDisambiguationProcessor(similarity_finder=finder, merger=merger)

        # when
        result = processor.process()

        # then
        assert len(result) == 1
        assert "node_similarity.cosine" in store.queries[0]
        assert any("refactor.merge_nodes" in q for q in store.queries)

    def test_neo4j_finder_with_neo4j_merger(self):
        # given
        store = FakeNeo4jStore(
            similarity_pairs=[
                {"name1": "X", "name2": "Y", "similarity": 0.95},
            ]
        )
        finder = Neo4jGDSEntitySimilarityFinder(source=store, embedding_attr="embedding")
        merger = Neo4jEntityMerger(source=store, preview_changes=False)
        processor = EntityDisambiguationProcessor(similarity_finder=finder, merger=merger)

        # when
        result = processor.process()

        # then
        assert len(result) == 1
        assert "vector.similarity.cosine" in store.queries[0]
        assert any("apoc.refactor.mergeNodes" in q for q in store.queries)

    def test_preview_mode_returns_empty(self):
        # given
        store = FakeMemgraphStore(
            similarity_pairs=[
                {"name1": "X", "name2": "Y", "similarity": 0.95},
            ]
        )
        finder = MemgraphCypherEntitySimilarityFinder(source=store, embedding_attr="embedding")
        merger = MemgraphEntityMerger(source=store, preview_changes=True)
        processor = EntityDisambiguationProcessor(similarity_finder=finder, merger=merger)

        # when
        result = processor.process()

        # then
        assert result == []

    def test_custom_finder_and_merger_composition(self):
        # given - use stubs to verify the processor delegates correctly
        class StubFinder:
            def find_similar_entities(self):
                return [{"Alpha", "Beta"}]

        class StubMerger:
            def merge_entities(self, groups):
                return [{"merged": g} for g in groups]

        processor = EntityDisambiguationProcessor(
            similarity_finder=StubFinder(),
            merger=StubMerger(),
        )

        # when
        result = processor.process()

        # then
        assert len(result) == 1
        assert result[0]["merged"] == {"Alpha", "Beta"}


class TestCrossBackendValidation:
    def test_memgraph_finder_rejects_neo4j_store(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Memgraph graph store"):
            MemgraphCypherEntitySimilarityFinder(source=store, embedding_attr="emb")

    def test_neo4j_finder_rejects_memgraph_store(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Neo4j graph store"):
            Neo4jGDSEntitySimilarityFinder(source=store, embedding_attr="emb")

    def test_memgraph_merger_rejects_neo4j_store(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Memgraph graph store"):
            MemgraphEntityMerger(source=store)

    def test_neo4j_merger_rejects_memgraph_store(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Neo4j graph store"):
            Neo4jEntityMerger(source=store)
