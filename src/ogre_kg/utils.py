"""Shared utilities for OGRE KG processors."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol


class GraphStoreBackend(Enum):
    """Supported property graph store backends."""

    MEMGRAPH = "memgraph"
    NEO4J = "neo4j"


class StructuredQueryCapableStore(Protocol):
    """Protocol for graph stores supporting structured query execution."""

    supports_structured_queries: bool

    def structured_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a synchronous backend query and return records."""

    async def astructured_query(self, query: str) -> list[dict[str, Any]]:
        """Execute an asynchronous backend query and return records."""


class PropertyGraphStoreProvider(Protocol):
    """Protocol for objects exposing a property graph store."""

    property_graph_store: StructuredQueryCapableStore


def resolve_graph_store(
    source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
) -> StructuredQueryCapableStore:
    """Resolve a graph store from either a store instance or index-like provider.

    Parameters
    ----------
    source
        Either a graph store object itself, or an object that exposes
        ``property_graph_store`` (for example PropertyGraphIndex).

    Returns
    -------
    StructuredQueryCapableStore
        Resolved graph store.

    Raises
    ------
    ValueError
        If source does not expose a property graph store interface.
    """
    if hasattr(source, "structured_query"):
        return source

    graph_store = getattr(source, "property_graph_store", None)
    if graph_store is not None and hasattr(graph_store, "structured_query"):
        return graph_store

    raise ValueError(
        "Expected a structured-query graph store or an index-like object with property_graph_store."
    )


def detect_graph_store_backend(store: Any) -> GraphStoreBackend:
    """Detect the backend type from a graph store instance.

    Inspects the class name and module path to determine whether the store
    is a Memgraph or Neo4j backend.

    Parameters
    ----------
    store
        A graph store instance.

    Returns
    -------
    GraphStoreBackend
        Detected backend type.

    Raises
    ------
    ValueError
        If the backend cannot be determined from the store type.
    """
    class_name = type(store).__name__
    module = type(store).__module__

    if "memgraph" in module.lower() or "Memgraph" in class_name:
        return GraphStoreBackend.MEMGRAPH
    if "neo4j" in module.lower() or "Neo4j" in class_name:
        return GraphStoreBackend.NEO4J

    raise ValueError(
        f"Cannot detect backend for store type '{class_name}' "
        f"from module '{module}'. Expected a Memgraph or Neo4j store."
    )


def quote_cypher(value: str) -> str:
    """Return Cypher-safe single-quoted string literal content.

    Parameters
    ----------
    value
        Raw string to be embedded in a Cypher query.

    Returns
    -------
    str
        Escaped string safe for inclusion inside single quotes.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


class UnionFind:
    """Union-find (disjoint set) data structure with path compression and union by rank.

    Parameters
    ----------
    elements
        Optional iterable of initial elements to add.
    """

    def __init__(self, elements: list[str] | None = None) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        if elements:
            for elem in elements:
                self._ensure_exists(elem)

    def _ensure_exists(self, node: str) -> None:
        """Add node to the structure if it does not already exist."""
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0

    def find(self, node: str) -> str:
        """Find the root representative of the set containing ``node``.

        Uses iterative path compression.

        Parameters
        ----------
        node
            Element to look up.

        Returns
        -------
        str
            Root representative of the set.
        """
        self._ensure_exists(node)

        # Walk to root
        root = node
        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression: point every node along the path directly to root
        current = node
        while current != root:
            next_parent = self._parent[current]
            self._parent[current] = root
            current = next_parent

        return root

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing ``a`` and ``b``.

        Uses union by rank.

        Parameters
        ----------
        a, b
            Elements whose sets should be merged.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        rank_a = self._rank[root_a]
        rank_b = self._rank[root_b]
        if rank_a < rank_b:
            self._parent[root_a] = root_b
        elif rank_a > rank_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] = rank_a + 1

    def groups(self, min_size: int = 2) -> list[set[str]]:
        """Return connected components as a list of sets.

        Parameters
        ----------
        min_size
            Minimum group size to include. Defaults to 2 (skip singletons).

        Returns
        -------
        list[set[str]]
            List of sets, each containing elements in the same component.
        """
        components: dict[str, set[str]] = {}
        for node in self._parent:
            root = self.find(node)
            if root not in components:
                components[root] = set()
            components[root].add(node)
        return [group for group in components.values() if len(group) >= min_size]


def build_entity_groups(pairs: list[dict[str, Any]]) -> list[set[str]]:
    """Build connected components from entity pair matches using union-find.

    Takes a list of pair dictionaries (each with ``name1`` and ``name2`` keys)
    and returns groups of transitively connected entity names.

    Parameters
    ----------
    pairs
        List of dicts with ``name1`` and ``name2`` string keys representing
        matched entity pairs.

    Returns
    -------
    list[set[str]]
        Connected components where each set contains >= 2 entity names.
    """
    uf = UnionFind()
    for pair in pairs:
        name1 = pair.get("name1")
        name2 = pair.get("name2")
        if isinstance(name1, str) and isinstance(name2, str) and name1 and name2:
            uf.union(name1, name2)
    return uf.groups(min_size=2)
