from __future__ import annotations

from dataclasses import dataclass

import pytest
from llama_index.core.schema import QueryBundle

from ogre_kg.kg_processors.retrievers import Keywords, MemgraphNodePathContextRetriever


class FakeLLM:
    def structured_predict(self, output_cls, prompt, question: str):
        assert output_cls is Keywords
        assert question == "Who collaborated with Tesla?"
        return Keywords(names=["Tesla", "Edison"])

    async def astructured_predict(self, output_cls, prompt, question: str):
        assert output_cls is Keywords
        assert question == "Who collaborated with Tesla?"
        return Keywords(names=["Tesla", "Edison"])


@dataclass
class FakeStructuredStore:
    supports_structured_queries: bool = True

    def __post_init__(self):
        self.queries: list[str] = []

    def structured_query(self, query: str):
        self.queries.append(query)
        return [{"source_text": "chunk", "matched_name": "entity", "score": 0.7}]

    async def astructured_query(self, query: str):
        self.queries.append(query)
        return [{"source_text": "chunk_async", "matched_name": "entity", "score": 0.9}]


def test_memgraph_retriever_sync_returns_nodes_for_each_keyword():
    # Given
    store = FakeStructuredStore()
    retriever = MemgraphNodePathContextRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=5,
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    # When
    nodes = retriever.retrieve_from_graph(query_bundle)

    # Then
    assert len(nodes) == 2
    assert "data.name:'Tesla'" in store.queries[0]
    assert "data.name:'Edison'" in store.queries[1]
    assert nodes[0].score == 0.7


@pytest.mark.asyncio
async def test_memgraph_retriever_async_returns_nodes_for_each_keyword():
    # Given
    store = FakeStructuredStore()
    retriever = MemgraphNodePathContextRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=2,
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    # When
    nodes = await retriever.aretrieve_from_graph(query_bundle)

    # Then
    assert len(nodes) == 2
    assert "data.name:'Tesla'" in store.queries[0]
    assert "data.name:'Edison'" in store.queries[1]
    assert nodes[0].score == 0.9


def test_memgraph_retriever_requires_structured_queries():
    # Given
    store = FakeStructuredStore(supports_structured_queries=False)

    # When / Then
    with pytest.raises(ValueError):
        MemgraphNodePathContextRetriever(
            graph_store=store,
            llm=FakeLLM(),
        )
