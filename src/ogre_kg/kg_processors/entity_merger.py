"""Entity group processors for property graph backends."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
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

            query = self.build_merge_query(group)

            if self.preview_changes:
                continue

            result = self.graph_store.structured_query(query)
            if result:
                merged_entities.append(result[0])

        return merged_entities


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
