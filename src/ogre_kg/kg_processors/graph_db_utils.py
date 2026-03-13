"""Utilities for building text/fulltext indices for graph backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ogre_kg.utils import (
    GraphStoreBackend,
    PropertyGraphStoreProvider,
    StructuredQueryCapableStore,
    detect_graph_store_backend,
    resolve_graph_store,
)


@dataclass(frozen=True)
class GraphTextIndexSpec:
    """Definition of one backend text-search index."""

    name: str
    label: str
    properties: tuple[str, ...]


def default_retriever_index_specs() -> list[GraphTextIndexSpec]:
    """Return default indices required by retriever text-search queries."""
    return [
        GraphTextIndexSpec(name="entity_name", label="__Entity__", properties=("name",)),
        GraphTextIndexSpec(name="chunk_text", label="Chunk", properties=("text",)),
    ]


class GraphTextIndexBuilder(ABC):
    """Base builder for backend-specific text/fulltext index creation."""

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
    ) -> None:
        self.graph_store = resolve_graph_store(source)

    @abstractmethod
    def build_create_query(
        self,
        spec: GraphTextIndexSpec,
        if_not_exists: bool = True,
    ) -> str:
        """Build a backend create-index query."""

    @abstractmethod
    def build_drop_query(
        self,
        index_name: str,
        if_exists: bool = True,
    ) -> str:
        """Build a backend drop-index query."""

    @staticmethod
    def _is_already_exists_error(exc: Exception) -> bool:
        """Return True when backend error indicates index already exists."""
        message = str(exc).lower()
        return "already exists" in message or "exists" in message

    def create_indices(
        self,
        specs: list[GraphTextIndexSpec] | None = None,
        if_not_exists: bool = True,
    ) -> None:
        """Create indices for provided specs."""
        for spec in specs or default_retriever_index_specs():
            query = self.build_create_query(spec, if_not_exists=if_not_exists)
            try:
                self.graph_store.structured_query(query)
            except Exception as exc:  # noqa: BLE001 - backend-specific error types vary
                if if_not_exists and self._is_already_exists_error(exc):
                    continue
                raise

    async def acreate_indices(
        self,
        specs: list[GraphTextIndexSpec] | None = None,
        if_not_exists: bool = True,
    ) -> None:
        """Asynchronously create indices for provided specs."""
        for spec in specs or default_retriever_index_specs():
            query = self.build_create_query(spec, if_not_exists=if_not_exists)
            try:
                await self.graph_store.astructured_query(query)
            except Exception as exc:  # noqa: BLE001 - backend-specific error types vary
                if if_not_exists and self._is_already_exists_error(exc):
                    continue
                raise

    def drop_indices(
        self,
        index_names: list[str],
        if_exists: bool = True,
    ) -> None:
        """Drop provided indices by name."""
        for index_name in index_names:
            query = self.build_drop_query(index_name=index_name, if_exists=if_exists)
            self.graph_store.structured_query(query)

    async def adrop_indices(
        self,
        index_names: list[str],
        if_exists: bool = True,
    ) -> None:
        """Asynchronously drop provided indices by name."""
        for index_name in index_names:
            query = self.build_drop_query(index_name=index_name, if_exists=if_exists)
            await self.graph_store.astructured_query(query)


class MemgraphTextIndexBuilder(GraphTextIndexBuilder):
    """Memgraph text index builder.

    Uses `CREATE TEXT INDEX ... ON :Label(property)` syntax.
    """

    ANALYTICAL_MODE_TEXT_INDEX_ERROR = "text index is not supported in analytical storage mode"
    INDEX_MULTICOMMAND_ERROR = "index manipulation not allowed in multicommand transactions"

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        auto_switch_to_transactional: bool = True,
    ) -> None:
        super().__init__(source=source)
        self.auto_switch_to_transactional = auto_switch_to_transactional

    def build_create_query(
        self,
        spec: GraphTextIndexSpec,
        if_not_exists: bool = True,
    ) -> str:
        del if_not_exists  # Memgraph text index creation does not expose IF NOT EXISTS.
        if len(spec.properties) != 1:
            raise ValueError(
                "Memgraph text index expects exactly one property per index. "
                f"Got {len(spec.properties)} for index '{spec.name}'."
            )

        prop = spec.properties[0]
        return f"CREATE TEXT INDEX {spec.name} ON :{spec.label}({prop});"

    def build_drop_query(
        self,
        index_name: str,
        if_exists: bool = True,
    ) -> str:
        del if_exists  # Memgraph text index drop does not expose IF EXISTS.
        return f"DROP TEXT INDEX {index_name};"

    def _run_autocommit_query(self, query: str) -> None:
        """Run query in an implicit auto-commit transaction when possible."""
        client = getattr(self.graph_store, "client", None)
        database = getattr(self.graph_store, "_database", None)

        if client is None or not hasattr(client, "session"):
            self.graph_store.structured_query(query)
            return

        with client.session(database=database) as session:
            result = session.run(query)
            # Materialize to ensure execution errors are raised immediately.
            list(result)

    def _ensure_transactional_storage_mode(self) -> None:
        """Switch Memgraph to transactional mode for text index operations."""
        self._run_autocommit_query("STORAGE MODE IN_MEMORY_TRANSACTIONAL;")

    def create_indices(
        self,
        specs: list[GraphTextIndexSpec] | None = None,
        if_not_exists: bool = True,
    ) -> None:
        for spec in specs or default_retriever_index_specs():
            query = self.build_create_query(spec, if_not_exists=if_not_exists)
            try:
                self._run_autocommit_query(query)
            except Exception as exc:  # noqa: BLE001 - backend-specific error types vary
                message = str(exc).lower()
                if if_not_exists and self._is_already_exists_error(exc):
                    continue

                if (
                    self.auto_switch_to_transactional
                    and self.ANALYTICAL_MODE_TEXT_INDEX_ERROR in message
                ):
                    self._ensure_transactional_storage_mode()
                    self._run_autocommit_query(query)
                    continue

                if self.INDEX_MULTICOMMAND_ERROR in message:
                    raise RuntimeError(
                        "Memgraph rejected index DDL in a multi-command transaction. "
                        "This builder uses auto-commit session.run() when driver session "
                        "access is available. Ensure you pass a Memgraph store instance "
                        "with driver client access."
                    ) from exc

                raise


class Neo4jFulltextIndexBuilder(GraphTextIndexBuilder):
    """Neo4j fulltext index builder.

    Uses `CREATE FULLTEXT INDEX ... FOR (n:Label) ON EACH [n.prop]` syntax.
    """

    def build_create_query(
        self,
        spec: GraphTextIndexSpec,
        if_not_exists: bool = True,
    ) -> str:
        props = ", ".join(f"n.{prop}" for prop in spec.properties)
        ine_clause = " IF NOT EXISTS" if if_not_exists else ""
        return (
            f"CREATE FULLTEXT INDEX {spec.name}{ine_clause} FOR (n:{spec.label}) ON EACH [{props}]"
        )

    def build_drop_query(
        self,
        index_name: str,
        if_exists: bool = True,
    ) -> str:
        if_exists_clause = " IF EXISTS" if if_exists else ""
        return f"DROP INDEX {index_name}{if_exists_clause}"


def make_graph_text_index_builder(
    source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
) -> GraphTextIndexBuilder:
    """Create backend-specific index builder for provided graph store."""
    graph_store = resolve_graph_store(source)
    backend = detect_graph_store_backend(graph_store)

    if backend == GraphStoreBackend.MEMGRAPH:
        return MemgraphTextIndexBuilder(source=graph_store)
    if backend == GraphStoreBackend.NEO4J:
        return Neo4jFulltextIndexBuilder(source=graph_store)

    raise ValueError(f"Unsupported graph backend: {backend.value}")
