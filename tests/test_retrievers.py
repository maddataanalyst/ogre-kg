from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.graph_stores.types import KG_SOURCE_REL, EntityNode, Relation
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle

from ogre_kg.kg_processors.retrievers import (
    MemgraphChunkKeywordRetriever,
    MemgraphKeywordContextRetriever,
    Neo4jChunkKeywordRetriever,
    Neo4jKeywordContextRetriever,
)
from ogre_kg.kg_processors.retrievers.base_retrievers import (
    DEFAULT_CHUNK_TERMS_PROMPT,
    DEFAULT_KEYWORD_PROMPT,
    ChunkTerms,
    Keywords,
)


class FakeLLM:
    def __init__(
        self,
        names: list[str] | None = None,
        chunk_terms: list[str] | None = None,
        keyword_response: Keywords | None = None,
        chunk_response: ChunkTerms | None = None,
    ) -> None:
        self.names = ["Tesla", "Edison"] if names is None else names
        self.chunk_terms = ["basal cell", "treatment"] if chunk_terms is None else chunk_terms
        self.keyword_response = keyword_response
        self.chunk_response = chunk_response
        self.keyword_questions: list[str] = []
        self.chunk_questions: list[str] = []
        self.keyword_prompts: list[PromptTemplate] = []
        self.chunk_prompts: list[PromptTemplate] = []

    def structured_predict(self, output_cls, prompt, **kwargs):
        question = kwargs["question"]

        if output_cls is Keywords:
            self.keyword_prompts.append(prompt)
            self.keyword_questions.append(question)
            if self.keyword_response is not None:
                return self.keyword_response
            max_keywords = int(kwargs["max_keywords"])
            return Keywords(names=self.names[:max_keywords])

        if output_cls is ChunkTerms:
            self.chunk_prompts.append(prompt)
            self.chunk_questions.append(question)
            if self.chunk_response is not None:
                return self.chunk_response
            max_terms = int(kwargs["max_terms"])
            return ChunkTerms(terms=self.chunk_terms[:max_terms])

        raise AssertionError(f"Unsupported output class: {output_cls}")

    async def astructured_predict(self, output_cls, prompt, **kwargs):
        return self.structured_predict(output_cls=output_cls, prompt=prompt, **kwargs)


@dataclass
class FakeLLMMetadata:
    is_function_calling_model: bool = True
    is_chat_model: bool = True


class FakeFallbackLLM:
    def __init__(self) -> None:
        self.metadata = FakeLLMMetadata()

    def structured_predict(self, output_cls, prompt, **kwargs):
        del output_cls, prompt, kwargs
        raise ValueError("Expected at least one tool call, but got 0 tool calls.")

    async def astructured_predict(self, output_cls, prompt, **kwargs):
        del output_cls, prompt, kwargs
        raise ValueError("Expected at least one tool call, but got 0 tool calls.")

    def _extend_messages(self, messages):
        return list(messages)

    def chat(self, messages, **kwargs):
        del kwargs
        prompt_text = "\n".join(message.content or "" for message in messages)
        if "chunk fulltext search" in prompt_text:
            content = '{"terms":[" basal cell ","Basal Cell","treatment"]}'
        else:
            content = '{"names":[" Tesla ","tesla","Edison"]}'
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content))

    async def achat(self, messages, **kwargs):
        return self.chat(messages=messages, **kwargs)


@dataclass
class FakeStructuredStore:
    supports_structured_queries: bool = True

    def __post_init__(self) -> None:
        self.queries: list[str] = []
        self.get_calls: list[list[str]] = []
        self.rel_map_calls: list[dict[str, Any]] = []

    def structured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del param_map
        self.queries.append(query)
        if "data.name:'Tesla'" in query:
            return [
                {"node_id": "entity_tesla", "score": 0.8},
                {"node_id": "entity_common", "score": 0.6},
            ]
        if "data.name:'Edison'" in query:
            return [
                {"node_id": "entity_common", "score": 0.4},
                {"node_id": "entity_edison", "score": 0.9},
            ]
        return []

    async def astructured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.structured_query(query=query, param_map=param_map)

    def get(
        self,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> list[EntityNode]:
        del properties
        ids = ids or []
        self.get_calls.append(ids)
        return [EntityNode(name=node_id, label="Entity") for node_id in ids]

    async def aget(
        self,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> list[EntityNode]:
        return self.get(properties=properties, ids=ids)

    def get_rel_map(
        self,
        graph_nodes: list[EntityNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[tuple[EntityNode, Relation, EntityNode]]:
        self.rel_map_calls.append(
            {
                "ids": [node.id for node in graph_nodes],
                "depth": depth,
                "limit": limit,
                "ignore_rels": ignore_rels or [],
            }
        )
        return [
            (
                EntityNode(
                    name="entity_common",
                    label="Entity",
                    properties={"triplet_source_id": "chunk-1"},
                ),
                Relation(
                    label="RELATED_TO",
                    source_id="entity_common",
                    target_id="entity_tesla",
                ),
                EntityNode(name="entity_tesla", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_edison",
                    label="Entity",
                    properties={"triplet_source_id": "chunk-2"},
                ),
                Relation(
                    label="MENTIONS",
                    source_id="entity_edison",
                    target_id="entity_other",
                ),
                EntityNode(name="entity_other", label="Entity"),
            ),
        ]

    async def aget_rel_map(
        self,
        graph_nodes: list[EntityNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[tuple[EntityNode, Relation, EntityNode]]:
        return self.get_rel_map(
            graph_nodes=graph_nodes,
            depth=depth,
            limit=limit,
            ignore_rels=ignore_rels,
        )


@dataclass
class FakeNeo4jStructuredStore(FakeStructuredStore):
    def structured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del param_map
        self.queries.append(query)
        if "queryNodes(\"entity_name\", 'Tesla'" in query:
            return [
                {"node_id": "entity_tesla", "score": 0.2},
                {"node_id": "entity_common", "score": 0.3},
            ]
        if "queryNodes(\"entity_name\", 'Edison'" in query:
            return [
                {"node_id": "entity_common", "score": 0.8},
                {"node_id": "entity_edison", "score": 0.1},
            ]
        return []


@dataclass
class FakeMemgraphChunkFirstStore(FakeStructuredStore):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.param_maps: list[dict[str, Any] | None] = []

    def structured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        self.param_maps.append(param_map)

        if 'text_search.search("chunk_text", $search_query, $topk)' in query:
            search_query = (param_map or {}).get("search_query", "")
            if "data.text:'basal cell'" in search_query:
                return [
                    {"node_id": "chunk_a", "score": 0.2},
                    {"node_id": "chunk_b", "score": 0.7},
                ]
            if "data.text:'treatment'" in search_query:
                return [
                    {"node_id": "chunk_a", "score": 0.1},
                    {"node_id": "chunk_c", "score": 0.5},
                ]
            return []

        if "MATCH (c:Chunk)-[r]->(e:__Entity__)" in query:
            chunk_ids = set((param_map or {}).get("chunk_ids", []))
            rows: list[dict[str, str]] = []
            if "chunk_a" in chunk_ids:
                rows.append({"node_id": "entity_shared", "chunk_id": "chunk_a"})
            if "chunk_b" in chunk_ids:
                rows.append({"node_id": "entity_shared", "chunk_id": "chunk_b"})
                rows.append({"node_id": "entity_from_b", "chunk_id": "chunk_b"})
            if "chunk_c" in chunk_ids:
                rows.append({"node_id": "entity_from_c", "chunk_id": "chunk_c"})
            return rows

        return []

    async def astructured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.structured_query(query=query, param_map=param_map)

    def get(
        self,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> list[EntityNode]:
        del properties
        ids = ids or []
        self.get_calls.append(ids)
        return [EntityNode(name=node_id, label="Entity") for node_id in ids]

    def get_rel_map(
        self,
        graph_nodes: list[EntityNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[tuple[EntityNode, Relation, EntityNode]]:
        self.rel_map_calls.append(
            {
                "ids": [node.id for node in graph_nodes],
                "depth": depth,
                "limit": limit,
                "ignore_rels": ignore_rels or [],
            }
        )
        return [
            (
                EntityNode(
                    name="entity_shared",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_a"},
                ),
                Relation(
                    label="RELATED_TO",
                    source_id="entity_shared",
                    target_id="entity_t",
                ),
                EntityNode(name="entity_t", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_from_b",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_b"},
                ),
                Relation(
                    label="ASSOCIATED_WITH",
                    source_id="entity_from_b",
                    target_id="entity_u",
                ),
                EntityNode(name="entity_u", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_from_c",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_c"},
                ),
                Relation(
                    label="LINKED_TO",
                    source_id="entity_from_c",
                    target_id="entity_w",
                ),
                EntityNode(name="entity_w", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_noise",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_noise"},
                ),
                Relation(
                    label="IRRELEVANT",
                    source_id="entity_noise",
                    target_id="entity_v",
                ),
                EntityNode(name="entity_v", label="Entity"),
            ),
        ]


@dataclass
class FakeNeo4jChunkFirstStore(FakeStructuredStore):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.param_maps: list[dict[str, Any] | None] = []

    def structured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        self.param_maps.append(param_map)

        if (
            "db.index.fulltext.queryNodes" in query
            and '"chunk_text", $search_query, {limit: $topk}' in query
        ):
            search_query = (param_map or {}).get("search_query", "")
            if search_query == "basal cell":
                return [
                    {"node_id": "chunk_a", "score": 0.6},
                    {"node_id": "chunk_b", "score": 0.3},
                ]
            if search_query == "treatment":
                return [
                    {"node_id": "chunk_a", "score": 0.9},
                    {"node_id": "chunk_c", "score": 0.4},
                ]
            return []

        if "MATCH (c:Chunk)-[r]->(e:__Entity__)" in query:
            chunk_ids = set((param_map or {}).get("chunk_ids", []))
            rows: list[dict[str, str]] = []
            if "chunk_a" in chunk_ids:
                rows.append({"node_id": "entity_shared", "chunk_id": "chunk_a"})
            if "chunk_b" in chunk_ids:
                rows.append({"node_id": "entity_shared", "chunk_id": "chunk_b"})
                rows.append({"node_id": "entity_from_b", "chunk_id": "chunk_b"})
            if "chunk_c" in chunk_ids:
                rows.append({"node_id": "entity_from_c", "chunk_id": "chunk_c"})
            return rows

        return []

    async def astructured_query(
        self,
        query: str,
        param_map: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.structured_query(query=query, param_map=param_map)

    def get(
        self,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> list[EntityNode]:
        del properties
        ids = ids or []
        self.get_calls.append(ids)
        return [EntityNode(name=node_id, label="Entity") for node_id in ids]

    def get_rel_map(
        self,
        graph_nodes: list[EntityNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[tuple[EntityNode, Relation, EntityNode]]:
        self.rel_map_calls.append(
            {
                "ids": [node.id for node in graph_nodes],
                "depth": depth,
                "limit": limit,
                "ignore_rels": ignore_rels or [],
            }
        )
        return [
            (
                EntityNode(
                    name="entity_shared",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_a"},
                ),
                Relation(
                    label="RELATED_TO",
                    source_id="entity_shared",
                    target_id="entity_t",
                ),
                EntityNode(name="entity_t", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_from_b",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_b"},
                ),
                Relation(
                    label="ASSOCIATED_WITH",
                    source_id="entity_from_b",
                    target_id="entity_u",
                ),
                EntityNode(name="entity_u", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_from_c",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_c"},
                ),
                Relation(
                    label="LINKED_TO",
                    source_id="entity_from_c",
                    target_id="entity_w",
                ),
                EntityNode(name="entity_w", label="Entity"),
            ),
            (
                EntityNode(
                    name="entity_noise",
                    label="Entity",
                    properties={"triplet_source_id": "chunk_noise"},
                ),
                Relation(
                    label="IRRELEVANT",
                    source_id="entity_noise",
                    target_id="entity_v",
                ),
                EntityNode(name="entity_v", label="Entity"),
            ),
        ]


def test_memgraph_retriever_sync_uses_seed_search_then_rel_map():
    store = FakeStructuredStore()
    retriever = MemgraphKeywordContextRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=5,
        path_depth=2,
        path_limit=11,
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert len(nodes) == 2
    assert "data.name:'Tesla'" in store.queries[0]
    assert "data.name:'Edison'" in store.queries[1]
    assert set(store.get_calls[0]) == {"entity_tesla", "entity_common", "entity_edison"}
    assert store.rel_map_calls[0]["depth"] == 2
    assert store.rel_map_calls[0]["limit"] == 11
    assert store.rel_map_calls[0]["ignore_rels"] == [KG_SOURCE_REL]
    assert nodes[0].score == 0.4
    assert nodes[0].node.text == "entity_common -> RELATED_TO -> entity_tesla"
    assert nodes[0].node.ref_doc_id == "chunk-1"
    assert nodes[1].score == 0.9
    assert nodes[1].node.text == "entity_edison -> MENTIONS -> entity_other"
    assert nodes[1].node.ref_doc_id == "chunk-2"


def test_memgraph_retriever_normalizes_structured_keyword_output():
    # given
    store = FakeStructuredStore()
    llm = FakeLLM(
        keyword_response=Keywords(names=[" Tesla ", "tesla", "Edison", "", "Nikola Tesla"])
    )
    retriever = MemgraphKeywordContextRetriever(graph_store=store, llm=llm, topk_search=2)

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="Who collaborated with Tesla?"))

    # then
    assert llm.keyword_questions == ["Who collaborated with Tesla?"]
    assert "data.name:'Tesla'" in store.queries[0]
    assert "data.name:'Edison'" in store.queries[1]
    assert len(store.queries) == 2


def test_memgraph_retriever_uses_default_keyword_prompt():
    # given
    store = FakeStructuredStore()
    llm = FakeLLM()
    retriever = MemgraphKeywordContextRetriever(graph_store=store, llm=llm)

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="Who collaborated with Tesla?"))

    # then
    assert llm.keyword_prompts == [DEFAULT_KEYWORD_PROMPT]


def test_memgraph_retriever_accepts_string_keyword_prompt_override():
    # given
    store = FakeStructuredStore()
    llm = FakeLLM()
    prompt_text = "Wyodrebnij encje z pytania.\nQUESTION: {question}"
    retriever = MemgraphKeywordContextRetriever(
        graph_store=store,
        llm=llm,
        keyword_prompt=prompt_text,
    )

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="Who collaborated with Tesla?"))

    # then
    assert llm.keyword_prompts[0].template == prompt_text


def test_memgraph_retriever_accepts_prompt_template_keyword_override():
    # given
    store = FakeStructuredStore()
    llm = FakeLLM()
    prompt = PromptTemplate("Extract graph entities.\nQUESTION: {question}")
    retriever = MemgraphKeywordContextRetriever(
        graph_store=store,
        llm=llm,
        keyword_prompt=prompt,
    )

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="Who collaborated with Tesla?"))

    # then
    assert llm.keyword_prompts == [prompt]


def test_memgraph_retriever_falls_back_when_function_calls_are_missing():
    # given
    store = FakeStructuredStore()
    retriever = MemgraphKeywordContextRetriever(
        graph_store=store,
        llm=FakeFallbackLLM(),
        topk_search=2,
    )

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="Who collaborated with Tesla?"))

    # then
    assert "data.name:'Tesla'" in store.queries[0]
    assert "data.name:'Edison'" in store.queries[1]
    assert len(store.queries) == 2


@pytest.mark.asyncio
async def test_memgraph_retriever_async_uses_seed_search_then_rel_map():
    store = FakeStructuredStore()
    retriever = MemgraphKeywordContextRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=2,
        path_depth=1,
        path_limit=7,
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    nodes = await retriever.aretrieve_from_graph(query_bundle)

    assert len(nodes) == 2
    assert "data.name:'Tesla'" in store.queries[0]
    assert "data.name:'Edison'" in store.queries[1]
    assert store.rel_map_calls[0]["depth"] == 1
    assert store.rel_map_calls[0]["limit"] == 7
    assert nodes[0].score == 0.4
    assert nodes[1].score == 0.9


def test_memgraph_retriever_escapes_keyword_for_cypher():
    store = FakeStructuredStore()
    retriever = MemgraphKeywordContextRetriever(
        graph_store=store,
        llm=FakeLLM(names=["O'Brien"]),
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    retriever.retrieve_from_graph(query_bundle)

    assert "data.name:'O\\'Brien'" in store.queries[0]


def test_memgraph_retriever_requires_structured_queries():
    store = FakeStructuredStore(supports_structured_queries=False)

    with pytest.raises(ValueError):
        MemgraphKeywordContextRetriever(
            graph_store=store,
            llm=FakeLLM(),
        )


def test_neo4j_retriever_uses_fulltext_seed_query_and_descending_scores():
    store = FakeNeo4jStructuredStore()
    retriever = Neo4jKeywordContextRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=4,
        path_depth=1,
        path_limit=9,
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert len(nodes) == 2
    assert "db.index.fulltext.queryNodes(\"entity_name\", 'Tesla', {limit: 4})" in store.queries[0]
    assert "db.index.fulltext.queryNodes(\"entity_name\", 'Edison', {limit: 4})" in store.queries[1]
    assert nodes[0].score == 0.8
    assert nodes[0].node.text == "entity_common -> RELATED_TO -> entity_tesla"
    assert nodes[1].score == 0.1


@pytest.mark.asyncio
async def test_neo4j_retriever_async_uses_fulltext_seed_query():
    store = FakeNeo4jStructuredStore()
    retriever = Neo4jKeywordContextRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=2,
    )
    query_bundle = QueryBundle(query_str="Who collaborated with Tesla?")

    nodes = await retriever.aretrieve_from_graph(query_bundle)

    assert len(nodes) == 2
    assert "db.index.fulltext.queryNodes(\"entity_name\", 'Tesla', {limit: 2})" in store.queries[0]
    assert "db.index.fulltext.queryNodes(\"entity_name\", 'Edison', {limit: 2})" in store.queries[1]
    assert nodes[0].score == 0.8


def test_memgraph_chunk_retriever_uses_terms_per_term_topk_and_chunk_dedup():
    store = FakeMemgraphChunkFirstStore()
    llm = FakeLLM(chunk_terms=["basal cell", "treatment", "ignored"])
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=llm,
        topk_search=3,
        max_chunk_terms=2,
        path_depth=2,
        path_limit=10,
        chunk_link_rels=("MENTIONS", "HAS_ENTITY"),
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert llm.chunk_questions == ["basal cell treatment options"]
    assert len(nodes) == 3
    assert store.queries[0].startswith(
        'CALL text_search.search("chunk_text", $search_query, $topk)'
    )
    assert store.param_maps[0] == {
        "search_query": "data.text:'basal cell'",
        "topk": 3,
    }
    assert store.param_maps[1] == {"search_query": "data.text:'treatment'", "topk": 3}
    assert store.param_maps[2] == {
        "chunk_ids": ["chunk_a", "chunk_b", "chunk_c"],
        "allowed_rels": ["MENTIONS", "HAS_ENTITY"],
    }
    assert store.rel_map_calls[0]["depth"] == 2
    assert store.rel_map_calls[0]["limit"] == 10
    assert store.rel_map_calls[0]["ignore_rels"] == [KG_SOURCE_REL]
    assert nodes[0].node.text == "entity_shared -> RELATED_TO -> entity_t"
    assert nodes[0].score == 0.1
    assert nodes[1].node.text == "entity_from_c -> LINKED_TO -> entity_w"
    assert nodes[1].score == 0.5
    assert nodes[2].node.text == "entity_from_b -> ASSOCIATED_WITH -> entity_u"
    assert nodes[2].score == 0.7


def test_memgraph_chunk_retriever_can_disable_chunk_source_filtering():
    store = FakeMemgraphChunkFirstStore()
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(),
        restrict_to_seed_chunks=False,
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert len(nodes) == 4
    assert nodes[3].node.text == "entity_noise -> IRRELEVANT -> entity_v"
    assert nodes[3].score == float("inf")


@pytest.mark.asyncio
async def test_memgraph_chunk_retriever_async_chunk_first_pipeline():
    store = FakeMemgraphChunkFirstStore()
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=2,
        chunk_link_rels=("MENTIONS",),
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = await retriever.aretrieve_from_graph(query_bundle)

    assert len(nodes) == 3
    assert store.param_maps[0] == {
        "search_query": "data.text:'basal cell'",
        "topk": 2,
    }
    assert store.param_maps[1] == {"search_query": "data.text:'treatment'", "topk": 2}
    assert store.param_maps[2] == {
        "chunk_ids": ["chunk_a", "chunk_b", "chunk_c"],
        "allowed_rels": ["MENTIONS"],
    }


def test_memgraph_chunk_retriever_returns_empty_when_no_chunk_terms():
    store = FakeMemgraphChunkFirstStore()
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(chunk_terms=[]),
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert nodes == []
    assert store.queries == []


def test_memgraph_chunk_retriever_normalizes_structured_chunk_terms():
    store = FakeMemgraphChunkFirstStore()
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(
            chunk_response=ChunkTerms(
                terms=[" basal cell ", "treatment", "", "Basal Cell", "ignored"]
            )
        ),
        max_chunk_terms=2,
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    retriever.retrieve_from_graph(query_bundle)

    assert store.param_maps[0] == {"search_query": "data.text:'basal cell'", "topk": 3}
    assert store.param_maps[1] == {"search_query": "data.text:'treatment'", "topk": 3}
    assert len(store.param_maps) == 3


def test_memgraph_chunk_retriever_uses_default_chunk_terms_prompt():
    # given
    store = FakeMemgraphChunkFirstStore()
    llm = FakeLLM()
    retriever = MemgraphChunkKeywordRetriever(graph_store=store, llm=llm)

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="basal cell treatment options"))

    # then
    assert llm.chunk_prompts == [DEFAULT_CHUNK_TERMS_PROMPT]


def test_memgraph_chunk_retriever_accepts_string_chunk_terms_prompt_override():
    # given
    store = FakeMemgraphChunkFirstStore()
    llm = FakeLLM()
    prompt_text = "Wygeneruj frazy wyszukiwania chunkow.\nQUESTION: {question}"
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=llm,
        chunk_terms_prompt=prompt_text,
    )

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="basal cell treatment options"))

    # then
    assert llm.chunk_prompts[0].template == prompt_text


def test_memgraph_chunk_retriever_accepts_prompt_template_chunk_terms_override():
    # given
    store = FakeMemgraphChunkFirstStore()
    llm = FakeLLM()
    prompt = PromptTemplate("Extract chunk search phrases.\nQUESTION: {question}")
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=llm,
        chunk_terms_prompt=prompt,
    )

    # when
    retriever.retrieve_from_graph(QueryBundle(query_str="basal cell treatment options"))

    # then
    assert llm.chunk_prompts == [prompt]


@pytest.mark.asyncio
async def test_memgraph_chunk_retriever_async_falls_back_when_function_calls_are_missing():
    store = FakeMemgraphChunkFirstStore()
    retriever = MemgraphChunkKeywordRetriever(
        graph_store=store,
        llm=FakeFallbackLLM(),
        max_chunk_terms=2,
    )

    nodes = await retriever.aretrieve_from_graph(QueryBundle(query_str="basal cell treatment"))

    assert len(nodes) == 3
    assert store.param_maps[0] == {"search_query": "data.text:'basal cell'", "topk": 3}
    assert store.param_maps[1] == {"search_query": "data.text:'treatment'", "topk": 3}


def test_memgraph_chunk_retriever_requires_non_empty_chunk_link_rels():
    store = FakeMemgraphChunkFirstStore()

    with pytest.raises(ValueError, match="chunk_link_rels"):
        MemgraphChunkKeywordRetriever(
            graph_store=store,
            llm=FakeLLM(),
            chunk_link_rels=(),
        )


def test_memgraph_chunk_retriever_requires_positive_max_chunk_terms():
    store = FakeMemgraphChunkFirstStore()

    with pytest.raises(ValueError, match="max_chunk_terms"):
        MemgraphChunkKeywordRetriever(
            graph_store=store,
            llm=FakeLLM(),
            max_chunk_terms=0,
        )


def test_neo4j_chunk_retriever_uses_terms_per_term_topk_and_chunk_dedup():
    store = FakeNeo4jChunkFirstStore()
    llm = FakeLLM(chunk_terms=["basal cell", "treatment"])
    retriever = Neo4jChunkKeywordRetriever(
        graph_store=store,
        llm=llm,
        topk_search=3,
        path_depth=2,
        path_limit=10,
        chunk_link_rels=("MENTIONS", "HAS_ENTITY"),
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert llm.chunk_questions == ["basal cell treatment options"]
    assert len(nodes) == 3
    assert store.queries[0].startswith("CALL db.index.fulltext.queryNodes(")
    assert '"chunk_text", $search_query, {limit: $topk}' in store.queries[0]
    assert store.param_maps[0] == {"search_query": "basal cell", "topk": 3}
    assert store.param_maps[1] == {"search_query": "treatment", "topk": 3}
    assert store.param_maps[2] == {
        "chunk_ids": ["chunk_a", "chunk_b", "chunk_c"],
        "allowed_rels": ["MENTIONS", "HAS_ENTITY"],
    }
    assert store.rel_map_calls[0]["depth"] == 2
    assert store.rel_map_calls[0]["limit"] == 10
    assert store.rel_map_calls[0]["ignore_rels"] == [KG_SOURCE_REL]
    assert nodes[0].node.text == "entity_shared -> RELATED_TO -> entity_t"
    assert nodes[0].score == 0.9
    assert nodes[1].node.text == "entity_from_c -> LINKED_TO -> entity_w"
    assert nodes[1].score == 0.4
    assert nodes[2].node.text == "entity_from_b -> ASSOCIATED_WITH -> entity_u"
    assert nodes[2].score == 0.3


def test_neo4j_chunk_retriever_can_disable_chunk_source_filtering():
    store = FakeNeo4jChunkFirstStore()
    retriever = Neo4jChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(),
        restrict_to_seed_chunks=False,
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert len(nodes) == 4
    assert nodes[3].node.text == "entity_noise -> IRRELEVANT -> entity_v"
    assert nodes[3].score == float("-inf")


@pytest.mark.asyncio
async def test_neo4j_chunk_retriever_async_chunk_first_pipeline():
    store = FakeNeo4jChunkFirstStore()
    retriever = Neo4jChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(),
        topk_search=2,
        chunk_link_rels=("MENTIONS",),
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = await retriever.aretrieve_from_graph(query_bundle)

    assert len(nodes) == 3
    assert store.param_maps[0] == {"search_query": "basal cell", "topk": 2}
    assert store.param_maps[1] == {"search_query": "treatment", "topk": 2}
    assert store.param_maps[2] == {
        "chunk_ids": ["chunk_a", "chunk_b", "chunk_c"],
        "allowed_rels": ["MENTIONS"],
    }


def test_neo4j_chunk_retriever_returns_empty_when_no_chunk_terms():
    store = FakeNeo4jChunkFirstStore()
    retriever = Neo4jChunkKeywordRetriever(
        graph_store=store,
        llm=FakeLLM(chunk_terms=[]),
    )
    query_bundle = QueryBundle(query_str="basal cell treatment options")

    nodes = retriever.retrieve_from_graph(query_bundle)

    assert nodes == []
    assert store.queries == []


def test_neo4j_chunk_retriever_requires_non_empty_chunk_link_rels():
    store = FakeNeo4jChunkFirstStore()

    with pytest.raises(ValueError, match="chunk_link_rels"):
        Neo4jChunkKeywordRetriever(
            graph_store=store,
            llm=FakeLLM(),
            chunk_link_rels=(),
        )
