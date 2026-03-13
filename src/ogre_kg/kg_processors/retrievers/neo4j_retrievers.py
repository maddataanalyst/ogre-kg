from __future__ import annotations

from typing import Any

from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.llms import LLM

from .base_retrievers import (
    GenericChunkKeywordContextRetriever,
    GenericKeywordContextRetriever,
)


class Neo4jKeywordContextRetriever(GenericKeywordContextRetriever):
    """Neo4j implementation of keyword-seeded relation-path retrieval."""

    CYPHER_QUERY = """CALL db.index.fulltext.queryNodes("entity_name", '{name}', {{limit: {topk}}})
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score DESC"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        **kwargs: Any,
    ) -> None:
        if not getattr(graph_store, "supports_structured_queries", False):
            raise ValueError("The provided graph store does not support structured queries.")

        super().__init__(
            graph_store=graph_store,
            search_query=self.CYPHER_QUERY,
            llm=llm,
            topk_search=topk_search,
            higher_score_is_better=True,
            **kwargs,
        )


class Neo4jChunkKeywordRetriever(GenericChunkKeywordContextRetriever):
    """Neo4j implementation of chunk-seeded relation-path retrieval."""

    CYPHER_QUERY = """CALL db.index.fulltext.queryNodes("chunk_text", '{name}', {{limit: {topk}}})
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score DESC"""

    CHUNK_SEARCH_QUERY = """CALL db.index.fulltext.queryNodes(
"chunk_text", $search_query, {limit: $topk}
)
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score DESC"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        max_chunk_terms: int = 6,
        chunk_link_rels: tuple[str, ...] = ("MENTIONS",),
        restrict_to_seed_chunks: bool = True,
        **kwargs: Any,
    ) -> None:
        if not getattr(graph_store, "supports_structured_queries", False):
            raise ValueError("The provided graph store does not support structured queries.")

        super().__init__(
            graph_store=graph_store,
            search_query=self.CYPHER_QUERY,
            llm=llm,
            topk_search=topk_search,
            higher_score_is_better=True,
            max_chunk_terms=max_chunk_terms,
            chunk_link_rels=chunk_link_rels,
            restrict_to_seed_chunks=restrict_to_seed_chunks,
            **kwargs,
        )

    def _fetch_chunk_seed_matches(self, term: str) -> list[dict[str, Any]]:
        return self._graph_store.structured_query(
            self.CHUNK_SEARCH_QUERY,
            param_map={"search_query": term, "topk": self.topk},
        )

    async def _afetch_chunk_seed_matches(self, term: str) -> list[dict[str, Any]]:
        return await self._graph_store.astructured_query(
            self.CHUNK_SEARCH_QUERY,
            param_map={"search_query": term, "topk": self.topk},
        )
