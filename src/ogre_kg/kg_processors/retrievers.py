from __future__ import annotations

from abc import abstractmethod
from typing import Any

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.settings import Settings


class Keywords(BaseModel):
    """Structured keyword candidates extracted from user query."""

    names: list[str] = Field(
        default_factory=list,
        description="Possible entity names or graph keywords related to the query.",
    )


class BaseGraphContextRetriever(BasePGRetriever):
    """Base retriever that resolves query keywords and fetches graph context.

    Subclasses can implement backend-specific context fetching logic, including
    structured-query backends (Memgraph, Neo4j) or in-memory graph traversal.
    """

    KEYWORD_PROMPT = PromptTemplate(
        "Extract entity names and graph-relevant keywords from the question. "
        "Return only concise candidates.\nQuestion: {question}"
    )

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize retriever.

        Parameters
        ----------
        graph_store
            Property graph store used by LlamaIndex retriever APIs.
        llm
            LLM used for keyword extraction. Defaults to global LlamaIndex settings.
        topk_search
            Maximum number of backend records to retrieve per keyword.
        **kwargs
            Additional retriever options.
        """
        self.llm = llm or Settings.llm
        self.topk = topk_search
        super().__init__(
            graph_store=graph_store,
            include_text=False,
            include_properties=False,
            **kwargs,
        )

    @staticmethod
    def _record_to_node(record: dict[str, Any]) -> NodeWithScore:
        """Convert a backend record into a LlamaIndex score-bearing text node."""
        score = float(record.get("score", 0.0))
        record_text = " | ".join(f"{key}: {value}" for key, value in record.items())
        return NodeWithScore(node=TextNode(text=record_text), score=score)

    @abstractmethod
    def _fetch_keyword_context(self, keyword: str) -> list[dict[str, Any]]:
        """Fetch backend records for one keyword."""

    @abstractmethod
    async def _afetch_keyword_context(self, keyword: str) -> list[dict[str, Any]]:
        """Fetch backend records for one keyword asynchronously."""

    def _extract_keywords(self, question: str) -> Keywords:
        """Extract keyword candidates from a question using structured prediction."""
        return self.llm.structured_predict(
            Keywords,
            self.KEYWORD_PROMPT,
            question=question,
        )

    async def _aextract_keywords(self, question: str) -> Keywords:
        """Asynchronously extract keyword candidates from a question."""
        return await self.llm.astructured_predict(
            Keywords,
            self.KEYWORD_PROMPT,
            question=question,
        )

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve graph-context nodes by expanding keywords from a user query."""
        response = self._extract_keywords(query_bundle.query_str)

        extracted_nodes: list[NodeWithScore] = []
        for keyword in response.names:
            for record in self._fetch_keyword_context(keyword):
                extracted_nodes.append(self._record_to_node(record))

        return extracted_nodes

    async def aretrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Asynchronously retrieve graph-context nodes from a user query."""
        response = await self._aextract_keywords(query_bundle.query_str)

        extracted_nodes: list[NodeWithScore] = []
        for keyword in response.names:
            for record in await self._afetch_keyword_context(keyword):
                extracted_nodes.append(self._record_to_node(record))

        return extracted_nodes


class MemgraphNodePathContextRetriever(BaseGraphContextRetriever):
    """Memgraph implementation of contextual entity-path retrieval."""

    CYPHER_QUERY = """CALL text_search.search("entity_name", "data.name:'{name}'", {topk})
YIELD node, score
MATCH (n2:Chunk)-[r:MENTIONS]->(node)-[r2]->(n3)
RETURN n2.text AS source_text, type(r) AS rel_1, node.name AS matched_name,
       type(r2) AS rel_2, n3.name AS target_name, score
ORDER BY score ASC"""

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
            llm=llm,
            topk_search=topk_search,
            **kwargs,
        )

    def _fetch_keyword_context(self, keyword: str) -> list[dict[str, Any]]:
        """Fetch Memgraph context rows for a single keyword."""
        query = self.CYPHER_QUERY.format(name=keyword, topk=self.topk)
        return self._graph_store.structured_query(query)

    async def _afetch_keyword_context(self, keyword: str) -> list[dict[str, Any]]:
        """Asynchronously fetch Memgraph context rows for a single keyword."""
        query = self.CYPHER_QUERY.format(name=keyword, topk=self.topk)
        return await self._graph_store.astructured_query(query)
