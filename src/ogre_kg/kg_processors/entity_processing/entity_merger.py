"""Entity group processors for property graph backends."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from itertools import combinations
from typing import Any, Literal

from ogre_kg.utils import (
    GraphStoreBackend,
    PropertyGraphStoreProvider,
    StructuredQueryCapableStore,
    detect_graph_store_backend,
    quote_cypher,
    resolve_graph_store,
)


class EntityMerger(ABC):
    """Base class for processing groups of duplicate entities in a property graph.

    Parameters
    ----------
    source
        Graph store or index-like object exposing ``property_graph_store``.
    merging_strategy
        Strategy used by backend merge procedure for node properties, if applicable.
    merge_relations
        Whether relationships should also be merged, if supported.
    preview_changes
        If True, build merge queries but do not execute them.
    """

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        merging_strategy: str | None = None,
        merge_relations: bool | None = True,
        preview_changes: bool = True,
    ) -> None:
        self.graph_store = resolve_graph_store(source)
        self.merging_strategy = merging_strategy
        self.merge_relations = merge_relations
        self.preview_changes = preview_changes

    @abstractmethod
    def build_merge_query(self, entity_group: set[str]) -> str:
        """Build a backend-specific query for one entity group.

        Parameters
        ----------
        entity_group
            Set of entity names to process.

        Returns
        -------
        str
            Cypher query string.
        """

    def _execute_group_merge(self, entity_group: set[str]) -> dict[str, Any] | None:
        """Execute a merge for a single group.

        Parameters
        ----------
        entity_group
            Set of entity names to process.

        Returns
        -------
        dict[str, Any] | None
            Backend result for the processed group.
        """
        query = self.build_merge_query(entity_group)
        result = self.graph_store.structured_query(query)
        if result:
            return result[0]
        return None

    def merge_entities(self, entity_groups: list[set[str]]) -> list[dict[str, Any]]:
        """Process provided entity groups and return backend outputs.

        Parameters
        ----------
        entity_groups
            List of entity name sets, each representing a group to merge.

        Returns
        -------
        list[dict[str, Any]]
            Backend results for each processed group (empty if ``preview_changes`` is True).
        """
        merged_entities: list[dict[str, Any]] = []

        for group in entity_groups:
            if len(group) < 2:
                continue

            if self.preview_changes:
                continue

            result = self._execute_group_merge(group)
            if result is not None:
                merged_entities.append(result)

        return merged_entities


class CanonicalEntitySelectionStrategy(Enum):
    """Strategies for selecting the canonical node during entity merge."""

    MOST_CONNECTED = "most_connected"
    ALPHABETICAL_NAME = "alphabetical_name"


def _cypher_literal(value: Any) -> str:
    """Serialize a Python value into a Cypher literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return f"'{quote_cypher(value)}'"
    if isinstance(value, list):
        return "[" + ", ".join(_cypher_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(f"{key}: {_cypher_literal(item)}" for key, item in sorted(value.items()))
        return "{" + pairs + "}"
    raise TypeError(f"Unsupported Cypher literal type: {type(value)!r}")


def _freeze_value(value: Any) -> Any:
    """Convert nested values into hashable structures for deduplication."""
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _validate_relation_name(relation_name: str) -> str:
    """Validate a Cypher relationship type name."""
    if not relation_name:
        raise ValueError("relation_name must be a non-empty string.")

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", relation_name):
        raise ValueError(
            "Invalid relation_name. Expected a valid Cypher relationship type name "
            "using only letters, numbers, and underscores."
        )

    return relation_name


class SynonymCreator(EntityMerger, ABC):
    """Base class for non-destructive entity linking via synonym relationships.

    Parameters
    ----------
    source
        Graph store or index-like object exposing ``property_graph_store``.
    relation_name
        Relationship type to create between similar entities.
    bidirectional
        If True, create both ``(a)-[:TYPE]->(b)`` and ``(b)-[:TYPE]->(a)``.
        If False, create one deterministic direction per entity pair.
    preview_changes
        If True, build merge queries but do not execute them.
    """

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        relation_name: str = "SIMILAR_TO",
        bidirectional: bool = False,
        preview_changes: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            merging_strategy=None,
            merge_relations=None,
            preview_changes=preview_changes,
        )
        self.relation_name = _validate_relation_name(relation_name)
        self.bidirectional = bidirectional

    def build_merge_query(self, entity_group: set[str]) -> str:
        """Build a Cypher query creating synonym relationships for one group."""
        sorted_entities = sorted(entity_group)
        match_clauses = ",".join(
            f"(n{i}:__Entity__ {{name: '{quote_cypher(name)}'}})"
            for i, name in enumerate(sorted_entities)
        )

        relationship_clauses: list[str] = []
        for left_idx, right_idx in combinations(range(len(sorted_entities)), 2):
            relationship_clauses.append(
                f"MERGE (n{left_idx})-[:{self.relation_name}]->(n{right_idx})"
            )
            if self.bidirectional:
                relationship_clauses.append(
                    f"MERGE (n{right_idx})-[:{self.relation_name}]->(n{left_idx})"
                )

        entities_literal = ", ".join(f"'{quote_cypher(name)}'" for name in sorted_entities)
        merge_block = "\n".join(relationship_clauses)

        return (
            f"MATCH {match_clauses}\n"
            f"{merge_block}\n"
            f"RETURN [{entities_literal}] AS entities, "
            f"'{self.relation_name}' AS relationship_type, "
            f"{str(self.bidirectional).lower()} AS bidirectional, "
            f"{len(relationship_clauses)} AS relationships_created"
        )


class MemgraphEntityMerger(EntityMerger):
    """Entity merger using Memgraph ``refactor.merge_nodes`` procedure.

    Validates at init time that the provided graph store is a Memgraph backend.

    Parameters
    ----------
    source
        Memgraph graph store or index-like provider.
    merging_strategy
        One of ``override``, ``combine``, or ``discard``.
    merge_relations
        Whether relationships should also be merged.
    preview_changes
        If True, build merge queries but do not execute them.
    """

    VALID_STRATEGIES = ("override", "combine", "discard")

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        merging_strategy: Literal["override", "combine", "discard"] = "override",
        merge_relations: bool = True,
        preview_changes: bool = True,
    ) -> None:
        if merging_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid merging strategy '{merging_strategy}' for Memgraph. "
                f"Expected one of: {self.VALID_STRATEGIES}"
            )
        super().__init__(
            source=source,
            merging_strategy=merging_strategy,
            merge_relations=merge_relations,
            preview_changes=preview_changes,
        )
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.MEMGRAPH:
            raise ValueError(
                f"MemgraphEntityMerger requires a Memgraph graph store, "
                f"but detected backend: {backend.value}"
            )

    def build_merge_query(self, entity_group: set[str]) -> str:
        """Build a Memgraph ``refactor.merge_nodes`` query.

        Parameters
        ----------
        entity_group
            Set of entity names to merge.

        Returns
        -------
        str
            Cypher query using Memgraph's ``refactor.merge_nodes`` procedure.
        """
        sorted_entities = sorted(entity_group)
        config = (
            f"{{ properties: '{self.merging_strategy}', "
            f"mergeRels: {str(self.merge_relations).lower()} }}"
        )

        match_clauses = ",".join(
            f"(n{i}: __Entity__ {{name: '{quote_cypher(name)}'}})"
            for i, name in enumerate(sorted_entities)
        )
        node_list = ",".join(f"n{i}" for i in range(len(sorted_entities)))

        return (
            f"MATCH {match_clauses}\n"
            f"CALL refactor.merge_nodes([{node_list}], {config})\n"
            f"YIELD node\n"
            f"RETURN node"
        )


class MemgraphSynonymCreator(SynonymCreator):
    """Synonym creator for Memgraph graph stores."""

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        relation_name: str = "SIMILAR_TO",
        bidirectional: bool = False,
        preview_changes: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            relation_name=relation_name,
            bidirectional=bidirectional,
            preview_changes=preview_changes,
        )
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.MEMGRAPH:
            raise ValueError(
                f"MemgraphSynonymCreator requires a Memgraph graph store, "
                f"but detected backend: {backend.value}"
            )


class FalkorDBSynonymCreator(SynonymCreator):
    """Synonym creator for FalkorDB graph stores."""

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        relation_name: str = "SIMILAR_TO",
        bidirectional: bool = False,
        preview_changes: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            relation_name=relation_name,
            bidirectional=bidirectional,
            preview_changes=preview_changes,
        )
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.FALKORDB:
            raise ValueError(
                f"FalkorDBSynonymCreator requires a FalkorDB graph store, "
                f"but detected backend: {backend.value}"
            )


class FalkorDBEntityMerger(EntityMerger):
    """Entity merger implemented in Python for FalkorDB graph stores.

    FalkorDB does not expose a built-in node merge procedure equivalent to
    Neo4j APOC or Memgraph refactor utilities, so this merger orchestrates the
    merge by reading node and relationship state, selecting a canonical node,
    replaying relationships onto that canonical node, and deleting duplicates.

    Parameters
    ----------
    source
        FalkorDB graph store or index-like provider.
    canonical_selection_strategy
        Strategy used to choose the winner node in each duplicate group.
    merge_relations
        Whether relationships should be rewired onto the canonical node.
    preview_changes
        If True, skip executing merge steps.
    """

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        canonical_selection_strategy: CanonicalEntitySelectionStrategy = (
            CanonicalEntitySelectionStrategy.MOST_CONNECTED
        ),
        merge_relations: bool = True,
        preview_changes: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            merging_strategy=None,
            merge_relations=merge_relations,
            preview_changes=preview_changes,
        )
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.FALKORDB:
            raise ValueError(
                f"FalkorDBEntityMerger requires a FalkorDB graph store, "
                f"but detected backend: {backend.value}"
            )
        self.canonical_selection_strategy = canonical_selection_strategy

    def build_merge_query(self, entity_group: set[str]) -> str:
        """Build a preview query for a FalkorDB merge group.

        Parameters
        ----------
        entity_group
            Set of entity names to merge.

        Returns
        -------
        str
            Preview query indicating which entities would be processed.
        """
        return (
            "// ogre_kg:falkordb_merge_preview\n"
            f"RETURN {_cypher_literal(sorted(entity_group))} AS entities, "
            f"'{self.canonical_selection_strategy.value}' AS canonical_selection_strategy"
        )

    def _execute_group_merge(self, entity_group: set[str]) -> dict[str, Any] | None:
        nodes = self._fetch_group_nodes(entity_group)
        if len(nodes) < 2:
            return None

        canonical = self._select_canonical_node(nodes)
        duplicate_nodes = [node for node in nodes if node["node_id"] != canonical["node_id"]]
        merged_labels = self._merge_labels(nodes)
        merged_properties = self._merge_properties(canonical, duplicate_nodes)

        relationship_rows: list[dict[str, Any]] = []
        relationships_created = 0
        if self.merge_relations:
            relationship_rows = self._collect_relationship_rows(duplicate_nodes)
            relationships_created = self._recreate_relationships(
                canonical=canonical,
                group_nodes=nodes,
                relationship_rows=relationship_rows,
            )

        self._update_canonical_node(
            canonical_node_id=canonical["node_id"],
            labels=merged_labels,
            properties=merged_properties,
        )
        self._delete_duplicate_nodes(duplicate_nodes)

        return {
            "canonical_name": canonical["name"],
            "canonical_node_id": canonical["node_id"],
            "merged_entities": sorted(node["name"] for node in nodes),
            "deleted_node_ids": [node["node_id"] for node in duplicate_nodes],
            "relationships_created": relationships_created,
            "merged_labels": sorted(merged_labels),
            "merged_properties": merged_properties,
        }

    def _fetch_group_nodes(self, entity_group: set[str]) -> list[dict[str, Any]]:
        query = (
            "// ogre_kg:falkordb_fetch_nodes\n"
            "MATCH (n:__Entity__)\n"
            f"WHERE n.name IN {_cypher_literal(sorted(entity_group))}\n"
            "RETURN id(n) AS node_id, n.name AS name, labels(n) AS labels, "
            "properties(n) AS properties, size((n)--()) AS degree\n"
            "ORDER BY n.name"
        )
        return self.graph_store.structured_query(query)

    def _select_canonical_node(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        if self.canonical_selection_strategy == CanonicalEntitySelectionStrategy.ALPHABETICAL_NAME:
            return min(nodes, key=lambda node: (node["name"], node["node_id"]))

        return min(
            nodes,
            key=lambda node: (
                -int(node.get("degree", 0)),
                -len(node.get("properties", {})),
                node.get("name", ""),
                int(node["node_id"]),
            ),
        )

    def _merge_labels(self, nodes: list[dict[str, Any]]) -> set[str]:
        labels: set[str] = set()
        for node in nodes:
            labels.update(node.get("labels", []))
        return labels

    def _merge_properties(
        self,
        canonical: dict[str, Any],
        duplicate_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        merged_properties = dict(canonical.get("properties", {}))
        for duplicate in duplicate_nodes:
            for key, value in duplicate.get("properties", {}).items():
                merged_properties.setdefault(key, value)
        return merged_properties

    def _collect_relationship_rows(
        self, duplicate_nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        duplicate_node_ids = [node["node_id"] for node in duplicate_nodes]
        outgoing_query = (
            "// ogre_kg:falkordb_fetch_outgoing_relationships\n"
            "MATCH (source)-[r]->(target)\n"
            f"WHERE id(source) IN {_cypher_literal(duplicate_node_ids)}\n"
            "RETURN id(source) AS source_id, id(target) AS target_id, type(r) AS rel_type, "
            "properties(r) AS rel_properties, 'outgoing' AS direction"
        )
        incoming_query = (
            "// ogre_kg:falkordb_fetch_incoming_relationships\n"
            "MATCH (source)-[r]->(target)\n"
            f"WHERE id(target) IN {_cypher_literal(duplicate_node_ids)}\n"
            "RETURN id(source) AS source_id, id(target) AS target_id, type(r) AS rel_type, "
            "properties(r) AS rel_properties, 'incoming' AS direction"
        )
        return self.graph_store.structured_query(
            outgoing_query
        ) + self.graph_store.structured_query(incoming_query)

    def _recreate_relationships(
        self,
        canonical: dict[str, Any],
        group_nodes: list[dict[str, Any]],
        relationship_rows: list[dict[str, Any]],
    ) -> int:
        group_node_ids = {node["node_id"] for node in group_nodes}
        canonical_id = canonical["node_id"]
        dedupe_keys: set[tuple[Any, ...]] = set()
        created = 0

        for row in relationship_rows:
            rel_type = _validate_relation_name(row["rel_type"])
            rel_properties = row.get("rel_properties", {})
            new_source_id = canonical_id if row["source_id"] in group_node_ids else row["source_id"]
            new_target_id = canonical_id if row["target_id"] in group_node_ids else row["target_id"]

            dedupe_key = (
                new_source_id,
                new_target_id,
                rel_type,
                _freeze_value(rel_properties),
            )
            if dedupe_key in dedupe_keys:
                continue
            dedupe_keys.add(dedupe_key)

            create_query = (
                "// ogre_kg:falkordb_create_relationship\n"
                f"MATCH (source), (target) "
                f"WHERE id(source) = {new_source_id} AND id(target) = {new_target_id}\n"
                f"MERGE (source)-[r:{rel_type}]->(target)\n"
                f"SET r += {_cypher_literal(rel_properties)}\n"
                "RETURN type(r) AS relationship_type"
            )
            self.graph_store.structured_query(create_query)
            created += 1

        return created

    def _update_canonical_node(
        self,
        canonical_node_id: int,
        labels: set[str],
        properties: dict[str, Any],
    ) -> None:
        label_suffix = "".join(f":{label}" for label in sorted(labels) if label != "__Entity__")
        query = (
            "// ogre_kg:falkordb_update_canonical_node\n"
            f"MATCH (canonical) WHERE id(canonical) = {canonical_node_id}\n"
            f"SET canonical:__Entity__{label_suffix}\n"
            f"SET canonical += {_cypher_literal(properties)}\n"
            "RETURN canonical"
        )
        self.graph_store.structured_query(query)

    def _delete_duplicate_nodes(self, duplicate_nodes: list[dict[str, Any]]) -> None:
        duplicate_ids = [node["node_id"] for node in duplicate_nodes]
        if not duplicate_ids:
            return
        query = (
            "// ogre_kg:falkordb_delete_duplicate_nodes\n"
            "MATCH (n)\n"
            f"WHERE id(n) IN {_cypher_literal(duplicate_ids)}\n"
            "DETACH DELETE n"
        )
        self.graph_store.structured_query(query)


class Neo4jEntityMerger(EntityMerger):
    """Entity merger using Neo4j APOC ``apoc.refactor.mergeNodes`` procedure.

    Validates at init time that the provided graph store is a Neo4j backend.

    Parameters
    ----------
    source
        Neo4j graph store or index-like provider.
    merging_strategy
        One of ``overwrite``, ``combine``, or ``discard``.
    merge_relations
        Whether relationships should also be merged.
    preview_changes
        If True, build merge queries but do not execute them.
    """

    VALID_STRATEGIES = ("overwrite", "combine", "discard")

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        merging_strategy: Literal["overwrite", "combine", "discard"] = "overwrite",
        merge_relations: bool = True,
        preview_changes: bool = True,
    ) -> None:
        if merging_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid merging strategy '{merging_strategy}' for Neo4j. "
                f"Expected one of: {self.VALID_STRATEGIES}"
            )
        super().__init__(
            source=source,
            merging_strategy=merging_strategy,
            merge_relations=merge_relations,
            preview_changes=preview_changes,
        )
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.NEO4J:
            raise ValueError(
                f"Neo4jEntityMerger requires a Neo4j graph store, "
                f"but detected backend: {backend.value}"
            )

    def build_merge_query(self, entity_group: set[str]) -> str:
        """Build a Neo4j APOC ``apoc.refactor.mergeNodes`` query.

        Parameters
        ----------
        entity_group
            Set of entity names to merge.

        Returns
        -------
        str
            Cypher query using Neo4j's APOC ``apoc.refactor.mergeNodes`` procedure.
        """
        sorted_entities = sorted(entity_group)
        config = (
            f"{{ properties: '{self.merging_strategy}', "
            f"mergeRels: {str(self.merge_relations).lower()} }}"
        )

        match_clauses = ",".join(
            f"(n{i}: __Entity__ {{name: '{quote_cypher(name)}'}})"
            for i, name in enumerate(sorted_entities)
        )
        node_list = ",".join(f"n{i}" for i in range(len(sorted_entities)))

        return (
            f"MATCH {match_clauses}\n"
            f"CALL apoc.refactor.mergeNodes([{node_list}], {config})\n"
            f"YIELD node\n"
            f"RETURN node"
        )


class Neo4jSynonymCreator(SynonymCreator):
    """Synonym creator for Neo4j graph stores."""

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        relation_name: str = "SIMILAR_TO",
        bidirectional: bool = False,
        preview_changes: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            relation_name=relation_name,
            bidirectional=bidirectional,
            preview_changes=preview_changes,
        )
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.NEO4J:
            raise ValueError(
                f"Neo4jSynonymCreator requires a Neo4j graph store, "
                f"but detected backend: {backend.value}"
            )
