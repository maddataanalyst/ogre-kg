from __future__ import annotations

from typing import Any

from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate

from .base_retrievers import (
    DEFAULT_CHUNK_TERMS_PROMPT,
    DEFAULT_KEYWORD_PROMPT,
    GenericChunkKeywordContextRetriever,
    GenericKeywordContextRetriever,
)


class FalkorDBKeywordContextRetriever(GenericKeywordContextRetriever):
    """FalkorDB implementation of keyword-seeded relation-path retrieval."""

    CYPHER_QUERY = """CALL db.idx.fulltext.queryNodes('__Entity__', $search_query)
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score DESC
LIMIT $topk"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        keyword_prompt: str | PromptTemplate = DEFAULT_KEYWORD_PROMPT,
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
            keyword_prompt=keyword_prompt,
            **kwargs,
        )

    def _fetch_keyword_seed_matches(self, keyword: str) -> list[dict[str, Any]]:
        return self._graph_store.structured_query(
            self.CYPHER_QUERY,
            param_map={"search_query": keyword, "topk": self.topk},
        )

    async def _afetch_keyword_seed_matches(self, keyword: str) -> list[dict[str, Any]]:
        return await self._graph_store.astructured_query(
            self.CYPHER_QUERY,
            param_map={"search_query": keyword, "topk": self.topk},
        )


class FalkorDBChunkKeywordRetriever(GenericChunkKeywordContextRetriever):
    """FalkorDB implementation of chunk-seeded relation-path retrieval."""

    CYPHER_QUERY = """CALL db.idx.fulltext.queryNodes('Chunk', $search_query)
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score DESC
LIMIT $topk"""

    CHUNK_SEARCH_QUERY = """CALL db.idx.fulltext.queryNodes('Chunk', $search_query)
YIELD node, score
RETURN node.id AS node_id, labels(node) AS labels, score
ORDER BY score DESC
LIMIT $topk"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        max_chunk_terms: int = 6,
        chunk_link_rels: tuple[str, ...] = ("MENTIONS",),
        restrict_to_seed_chunks: bool = True,
        keyword_prompt: str | PromptTemplate = DEFAULT_KEYWORD_PROMPT,
        chunk_terms_prompt: str | PromptTemplate = DEFAULT_CHUNK_TERMS_PROMPT,
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
            keyword_prompt=keyword_prompt,
            chunk_terms_prompt=chunk_terms_prompt,
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
