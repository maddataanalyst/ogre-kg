"""Tests for ogre_kg.kg_processors.entity_processing.entity_merger module."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from conftest import FakeFalkorDBStore, FakeMemgraphStore, FakeNeo4jStore, FakeStructuredStore

from ogre_kg.kg_processors.entity_processing.entity_merger import (
    CanonicalEntitySelectionStrategy,
    FalkorDBEntityMerger,
    FalkorDBSynonymCreator,
    MemgraphEntityMerger,
    MemgraphSynonymCreator,
    Neo4jEntityMerger,
    Neo4jSynonymCreator,
)


@dataclass
class FakeFalkorDBMergeStore(FakeFalkorDBStore):
    nodes_by_name: dict[str, dict] = field(default_factory=dict)
    outgoing_relationships: list[dict] = field(default_factory=list)
    incoming_relationships: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.created_relationships: list[dict] = []
        self.updated_nodes: list[dict] = []
        self.deleted_node_ids: list[int] = []

    def structured_query(self, query: str) -> list[dict]:
        self.queries.append(query)

        if "ogre_kg:falkordb_fetch_nodes" in query:
            requested = [
                node for name, node in sorted(self.nodes_by_name.items()) if f"'{name}'" in query
            ]
            return [
                {
                    "node_id": node["node_id"],
                    "name": node["name"],
                    "labels": list(node.get("labels", [])),
                    "properties": dict(node.get("properties", {})),
                    "degree": node.get("degree", 0),
                }
                for node in requested
            ]

        if "ogre_kg:falkordb_fetch_outgoing_relationships" in query:
            return list(self.outgoing_relationships)

        if "ogre_kg:falkordb_fetch_incoming_relationships" in query:
            return list(self.incoming_relationships)

        if "ogre_kg:falkordb_create_relationship" in query:
            self.created_relationships.append({"query": query})
            return [{"relationship_type": "RELATED_TO"}]

        if "ogre_kg:falkordb_update_canonical_node" in query:
            self.updated_nodes.append({"query": query})
            return [{"canonical": {"updated": True}}]

        if "ogre_kg:falkordb_delete_duplicate_nodes" in query:
            for node in self.nodes_by_name.values():
                node_id = node["node_id"]
                if str(node_id) in query:
                    self.deleted_node_ids.append(node_id)
            return []

        return super().structured_query(query)


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


class TestMemgraphSynonymCreator:
    def test_build_merge_query_uses_similar_to_by_default(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store)

        # when
        query = merger.build_merge_query({"Alice", "Bob"})

        # then
        assert "MERGE (n0)-[:SIMILAR_TO]->(n1)" in query
        assert "relationship_type" in query

    def test_build_merge_query_respects_custom_relation_name(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store, relation_name="ALIAS_OF")

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "[:ALIAS_OF]" in query

    def test_build_merge_query_can_create_bidirectional_edges(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store, bidirectional=True)

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "MERGE (n0)-[:SIMILAR_TO]->(n1)" in query
        assert "MERGE (n1)-[:SIMILAR_TO]->(n0)" in query
        assert "2 AS relationships_created" in query

    def test_build_merge_query_creates_all_pairs_for_groups(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store)

        # when
        query = merger.build_merge_query({"Zebra", "Apple", "Mango"})

        # then
        assert query.count("MERGE (") == 3
        assert "3 AS relationships_created" in query

    def test_merge_entities_preview_mode_returns_empty(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store, preview_changes=True)

        # when
        result = merger.merge_entities([{"A", "B"}, {"C", "D"}])

        # then
        assert result == []
        assert len(store.queries) == 0

    def test_merge_entities_executes_when_preview_disabled(self):
        # given
        store = FakeMemgraphStore(
            synonym_result=[
                {
                    "entities": ["A", "B"],
                    "relationship_type": "SIMILAR_TO",
                    "bidirectional": False,
                    "relationships_created": 1,
                }
            ]
        )
        merger = MemgraphSynonymCreator(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == [
            {
                "entities": ["A", "B"],
                "relationship_type": "SIMILAR_TO",
                "bidirectional": False,
                "relationships_created": 1,
            }
        ]
        assert any("[:SIMILAR_TO]" in q for q in store.queries)

    def test_merge_entities_skips_single_element_groups(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A"}])

        # then
        assert result == []
        assert len(store.queries) == 0

    def test_special_characters_escaped_in_query(self):
        # given
        store = FakeMemgraphStore()
        merger = MemgraphSynonymCreator(source=store)

        # when
        query = merger.build_merge_query({"O'Brien", "McDonald's"})

        # then
        assert "O\\'Brien" in query
        assert "McDonald\\'s" in query

    def test_invalid_relation_name_raises_value_error(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="Invalid relation_name"):
            MemgraphSynonymCreator(source=store, relation_name="SIMILAR-TO")

    def test_rejects_neo4j_store(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Memgraph graph store"):
            MemgraphSynonymCreator(source=store)

    def test_rejects_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            MemgraphSynonymCreator(source=store)


class TestFalkorDBSynonymCreator:
    def test_build_merge_query_uses_similar_to_by_default(self):
        # given
        store = FakeFalkorDBStore()
        merger = FalkorDBSynonymCreator(source=store)

        # when
        query = merger.build_merge_query({"Alice", "Bob"})

        # then
        assert "MERGE (n0)-[:SIMILAR_TO]->(n1)" in query
        assert "relationship_type" in query

    def test_merge_entities_executes_when_preview_disabled(self):
        # given
        store = FakeFalkorDBStore(
            synonym_result=[
                {
                    "entities": ["A", "B"],
                    "relationship_type": "SIMILAR_TO",
                    "bidirectional": False,
                    "relationships_created": 1,
                }
            ]
        )
        merger = FalkorDBSynonymCreator(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == [
            {
                "entities": ["A", "B"],
                "relationship_type": "SIMILAR_TO",
                "bidirectional": False,
                "relationships_created": 1,
            }
        ]
        assert any("[:SIMILAR_TO]" in q for q in store.queries)

    def test_rejects_neo4j_store(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="requires a FalkorDB graph store"):
            FalkorDBSynonymCreator(source=store)


class TestFalkorDBEntityMerger:
    def test_preview_mode_returns_empty_without_queries(self):
        # given
        store = FakeFalkorDBMergeStore(
            nodes_by_name={
                "A": {
                    "node_id": 1,
                    "name": "A",
                    "labels": ["__Entity__"],
                    "properties": {},
                    "degree": 1,
                },
                "B": {
                    "node_id": 2,
                    "name": "B",
                    "labels": ["__Entity__"],
                    "properties": {},
                    "degree": 1,
                },
            }
        )
        merger = FalkorDBEntityMerger(source=store, preview_changes=True)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == []
        assert store.queries == []

    def test_default_strategy_selects_most_connected_node(self):
        # given
        store = FakeFalkorDBMergeStore(
            nodes_by_name={
                "A": {
                    "node_id": 1,
                    "name": "A",
                    "labels": ["__Entity__", "Alias"],
                    "properties": {"name": "A", "source": "db1"},
                    "degree": 5,
                },
                "B": {
                    "node_id": 2,
                    "name": "B",
                    "labels": ["__Entity__", "Canonical"],
                    "properties": {"name": "B", "source": "db2", "description": "winner"},
                    "degree": 8,
                },
            },
            outgoing_relationships=[
                {
                    "source_id": 1,
                    "target_id": 99,
                    "rel_type": "RELATED_TO",
                    "rel_properties": {"score": 0.5},
                    "direction": "outgoing",
                }
            ],
            incoming_relationships=[
                {
                    "source_id": 77,
                    "target_id": 1,
                    "rel_type": "MENTIONS",
                    "rel_properties": {},
                    "direction": "incoming",
                }
            ],
        )
        merger = FalkorDBEntityMerger(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == [
            {
                "canonical_name": "B",
                "canonical_node_id": 2,
                "merged_entities": ["A", "B"],
                "deleted_node_ids": [1],
                "relationships_created": 2,
                "merged_labels": ["Alias", "Canonical", "__Entity__"],
                "merged_properties": {
                    "name": "B",
                    "source": "db2",
                    "description": "winner",
                },
            }
        ]
        assert len(store.created_relationships) == 2
        assert store.deleted_node_ids == [1]
        assert "SET canonical:__Entity__:Alias:Canonical" in store.updated_nodes[0]["query"]
        assert "description: 'winner'" in store.updated_nodes[0]["query"]

    def test_can_select_canonical_by_alphabetical_name(self):
        # given
        store = FakeFalkorDBMergeStore(
            nodes_by_name={
                "Zebra": {
                    "node_id": 10,
                    "name": "Zebra",
                    "labels": ["__Entity__"],
                    "properties": {"name": "Zebra", "country": "ZA"},
                    "degree": 10,
                },
                "Apple": {
                    "node_id": 11,
                    "name": "Apple",
                    "labels": ["__Entity__"],
                    "properties": {"name": "Apple"},
                    "degree": 1,
                },
            }
        )
        merger = FalkorDBEntityMerger(
            source=store,
            canonical_selection_strategy=CanonicalEntitySelectionStrategy.ALPHABETICAL_NAME,
            preview_changes=False,
        )

        # when
        result = merger.merge_entities([{"Apple", "Zebra"}])

        # then
        assert result[0]["canonical_name"] == "Apple"
        assert result[0]["deleted_node_ids"] == [10]
        assert "country: 'ZA'" in store.updated_nodes[0]["query"]

    def test_tie_breaker_prefers_more_properties_then_name(self):
        # given
        store = FakeFalkorDBMergeStore(
            nodes_by_name={
                "Bravo": {
                    "node_id": 20,
                    "name": "Bravo",
                    "labels": ["__Entity__"],
                    "properties": {"name": "Bravo", "x": 1, "y": 2},
                    "degree": 3,
                },
                "Alpha": {
                    "node_id": 21,
                    "name": "Alpha",
                    "labels": ["__Entity__"],
                    "properties": {"name": "Alpha", "x": 1, "y": 2},
                    "degree": 3,
                },
            }
        )
        merger = FalkorDBEntityMerger(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"Alpha", "Bravo"}])

        # then
        assert result[0]["canonical_name"] == "Alpha"

    def test_deduplicates_rewritten_relationships_and_allows_self_loops(self):
        # given
        store = FakeFalkorDBMergeStore(
            nodes_by_name={
                "A": {
                    "node_id": 1,
                    "name": "A",
                    "labels": ["__Entity__"],
                    "properties": {"name": "A"},
                    "degree": 5,
                },
                "B": {
                    "node_id": 2,
                    "name": "B",
                    "labels": ["__Entity__"],
                    "properties": {"name": "B"},
                    "degree": 4,
                },
            },
            outgoing_relationships=[
                {
                    "source_id": 2,
                    "target_id": 99,
                    "rel_type": "RELATED_TO",
                    "rel_properties": {"score": 1},
                    "direction": "outgoing",
                },
                {
                    "source_id": 2,
                    "target_id": 99,
                    "rel_type": "RELATED_TO",
                    "rel_properties": {"score": 1},
                    "direction": "outgoing",
                },
                {
                    "source_id": 2,
                    "target_id": 1,
                    "rel_type": "SAME_AS",
                    "rel_properties": {},
                    "direction": "outgoing",
                },
            ],
        )
        merger = FalkorDBEntityMerger(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result[0]["relationships_created"] == 2
        assert len(store.created_relationships) == 2
        assert any(
            "MERGE (source)-[r:SAME_AS]->(target)" in rel["query"]
            for rel in store.created_relationships
        )

    def test_can_skip_relationship_rewiring(self):
        # given
        store = FakeFalkorDBMergeStore(
            nodes_by_name={
                "A": {
                    "node_id": 1,
                    "name": "A",
                    "labels": ["__Entity__"],
                    "properties": {"name": "A"},
                    "degree": 2,
                },
                "B": {
                    "node_id": 2,
                    "name": "B",
                    "labels": ["__Entity__"],
                    "properties": {"name": "B"},
                    "degree": 1,
                },
            },
            outgoing_relationships=[
                {
                    "source_id": 2,
                    "target_id": 99,
                    "rel_type": "RELATED_TO",
                    "rel_properties": {},
                    "direction": "outgoing",
                }
            ],
        )
        merger = FalkorDBEntityMerger(source=store, merge_relations=False, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result[0]["relationships_created"] == 0
        assert store.created_relationships == []

    def test_rejects_memgraph_store(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="requires a FalkorDB graph store"):
            FalkorDBEntityMerger(source=store)


class TestNeo4jSynonymCreator:
    def test_build_merge_query_uses_similar_to_by_default(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jSynonymCreator(source=store)

        # when
        query = merger.build_merge_query({"Alice", "Bob"})

        # then
        assert "MERGE (n0)-[:SIMILAR_TO]->(n1)" in query

    def test_build_merge_query_respects_custom_relation_name(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jSynonymCreator(source=store, relation_name="ALIAS_OF")

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "[:ALIAS_OF]" in query

    def test_build_merge_query_can_create_bidirectional_edges(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jSynonymCreator(source=store, bidirectional=True)

        # when
        query = merger.build_merge_query({"A", "B"})

        # then
        assert "MERGE (n0)-[:SIMILAR_TO]->(n1)" in query
        assert "MERGE (n1)-[:SIMILAR_TO]->(n0)" in query
        assert "2 AS relationships_created" in query

    def test_merge_entities_executes_when_preview_disabled(self):
        # given
        store = FakeNeo4jStore(
            synonym_result=[
                {
                    "entities": ["A", "B"],
                    "relationship_type": "SIMILAR_TO",
                    "bidirectional": False,
                    "relationships_created": 1,
                }
            ]
        )
        merger = Neo4jSynonymCreator(source=store, preview_changes=False)

        # when
        result = merger.merge_entities([{"A", "B"}])

        # then
        assert result == [
            {
                "entities": ["A", "B"],
                "relationship_type": "SIMILAR_TO",
                "bidirectional": False,
                "relationships_created": 1,
            }
        ]
        assert any("[:SIMILAR_TO]" in q for q in store.queries)

    def test_entities_sorted_for_deterministic_queries(self):
        # given
        store = FakeNeo4jStore()
        merger = Neo4jSynonymCreator(source=store)

        # when
        query = merger.build_merge_query({"Zebra", "Apple", "Mango"})

        # then
        apple_pos = query.index("Apple")
        mango_pos = query.index("Mango")
        zebra_pos = query.index("Zebra")
        assert apple_pos < mango_pos < zebra_pos

    def test_invalid_relation_name_raises_value_error(self):
        # given
        store = FakeNeo4jStore()

        # when / then
        with pytest.raises(ValueError, match="Invalid relation_name"):
            Neo4jSynonymCreator(source=store, relation_name="SIMILAR TO")

    def test_rejects_memgraph_store(self):
        # given
        store = FakeMemgraphStore()

        # when / then
        with pytest.raises(ValueError, match="requires a Neo4j graph store"):
            Neo4jSynonymCreator(source=store)

    def test_rejects_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            Neo4jSynonymCreator(source=store)
