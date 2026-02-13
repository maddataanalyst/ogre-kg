"""Tests for ogre_kg.kg_processors.entity_merger module."""

from __future__ import annotations

import pytest
from conftest import FakeMemgraphStore, FakeNeo4jStore, FakeStructuredStore

from ogre_kg.kg_processors.entity_merger import (
    MemgraphEntityMerger,
    Neo4jEntityMerger,
)


class TestMemgraphEntityMerger:
    def test_build_merge_query_uses_refactor_merge_nodes(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store)

        # when
        query = merger.build_merge_query({"Alice", "Bob"})

        # then
        assert "refactor.merge_nodes" in query
        assert "Alice" in query
        assert "Bob" in query

    def test_build_merge_query_uses_override_strategy(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store, merging_strategy="combine")

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "properties: 'combine'" in query

    def test_build_merge_query_respects_merge_relations(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store, merge_relations=False)

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "mergeRels: false" in query

    def test_merge_entities_preview_mode_returns_empty(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store, preview_changes=True)

        # when
        result = merger.merge_entities([{"A", "B"}, {"C", "D"}])

        # then
        assert result == []
        assert len(store.queries) == 0

    def test_merge_entities_executes_when_preview_disabled(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == [{"node": {"name": "merged"}}]
        assert any("refactor.merge_nodes" in q for q in store.queries)

    def test_merge_entities_skips_single_element_groups(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A"}])

        # then
        assert result == []
        assert len(store.queries) == 0

    def test_special_characters_escaped_in_query(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphEntityMerger(source=store)

        # when
        query = merger.build_merge_query({"O'Brien", "McDonald's"})

        # then
        assert "O\\'Brien" in query
        assert "McDonald\\'s" in query

    def test_invalid_strategy_raises_value_error(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="Invalid merging strategy"):
            MemgraphEntityMerger(source=store, merging_strategy="overwrite")

    def test_rejects_neo4j_store(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Memgraph graph store"):
            MemgraphEntityMerger(source=store)

    def test_rejects_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            MemgraphEntityMerger(source=store)


class TestNeo4jEntityMerger:
    def test_build_merge_query_uses_apoc(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jEntityMerger(source=store)

        # when
        query = merger.build_merge_query({"Alice", "Bob"})

        # then
        assert "apoc.refactor.mergeNodes" in query
        assert "Alice" in query
        assert "Bob" in query

    def test_default_strategy_is_overwrite(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jEntityMerger(source=store)

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "properties: 'overwrite'" in query

    def test_merge_entities_executes_when_preview_disabled(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jEntityMerger(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == [{"node": {"name": "merged"}}]
        assert any("apoc.refactor.mergeNodes" in q for q in store.queries)

    def test_invalid_strategy_raises_value_error(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="Invalid merging strategy"):
            Neo4jEntityMerger(source=store, merging_strategy="override")

    def test_entities_sorted_for_deterministic_queries(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jEntityMerger(source=store)

        # when
        query = merger.build_merge_query({"Zebra", "Apple", "Mango"})

        # then - entities should be sorted alphabetically
        apple_pos = query.index("Apple")
        mango_pos = query.index("Mango")
        zebra_pos = query.index("Zebra")
        assert apple_pos < mango_pos < zebra_pos

    def test_rejects_memgraph_store(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Neo4j graph store"):
            Neo4jEntityMerger(source=store)

    def test_rejects_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            Neo4jEntityMerger(source=store)
