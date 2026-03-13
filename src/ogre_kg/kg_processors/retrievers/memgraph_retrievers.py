from __future__ import annotations

from typing import Any

from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.llms import LLM

from ogre_kg.utils import quote_cypher

from .base_retrievers import (
    GenericChunkKeywordContextRetriever,
    GenericKeywordContextRetriever,
)


class MemgraphKeywordContextRetriever(GenericKeywordContextRetriever):
    """Memgraph implementation of keyword-seeded relation-path retrieval."""

    CYPHER_QUERY = """CALL text_search.search("entity_name", "data.name:'{name}'", {topk})
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score ASC"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        higher_score_is_better: bool = False,
        **kwargs: Any,
    ) -> None:
        if not getattr(graph_store, "supports_structured_queries", False):
            raise ValueError("The provided graph store does not support structured queries.")

        super().__init__(
            graph_store=graph_store,
            search_query=self.CYPHER_QUERY,
            llm=llm,
            topk_search=topk_search,
            higher_score_is_better=higher_score_is_better,
            **kwargs,
        )


class MemgraphChunkKeywordRetriever(GenericChunkKeywordContextRetriever):
    CYPHER_QUERY = """CALL text_search.search("chunk_text", "data.text:'{name}'", {topk})
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score ASC"""

    CHUNK_SEARCH_QUERY = """CALL text_search.search("chunk_text", $search_query, $topk)
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score ASC"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        max_chunk_terms: int = 6,
        chunk_link_rels: tuple[str, ...] = ("MENTIONS",),
        restrict_to_seed_chunks: bool = True,
        higher_score_is_better: bool = False,
        **kwargs: Any,
    ) -> None:
        if not getattr(graph_store, "supports_structured_queries", False):
            raise ValueError("The provided graph store does not support structured queries.")

        super().__init__(
            graph_store=graph_store,
            search_query=self.CYPHER_QUERY,
            llm=llm,
            topk_search=topk_search,
            higher_score_is_better=higher_score_is_better,
            max_chunk_terms=max_chunk_terms,
            chunk_link_rels=chunk_link_rels,
            restrict_to_seed_chunks=restrict_to_seed_chunks,
            **kwargs,
        )

    def _build_chunk_text_search(self, term: str) -> str:
        return f"data.text:'{quote_cypher(term)}'"

    def _fetch_chunk_seed_matches(self, term: str) -> list[dict[str, Any]]:
        return self._graph_store.structured_query(
            self.CHUNK_SEARCH_QUERY,
            param_map={
                "search_query": self._build_chunk_text_search(term),
                "topk": self.topk,
            },
        )

    async def _afetch_chunk_seed_matches(self, term: str) -> list[dict[str, Any]]:
        return await self._graph_store.astructured_query(
            self.CHUNK_SEARCH_QUERY,
            param_map={
                "search_query": self._build_chunk_text_search(term),
                "topk": self.topk,
            },
        )
