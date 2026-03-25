from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

from llama_index.core.graph_stores.types import KG_SOURCE_REL, PropertyGraphStore, Triplet
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.llms import LLM
from llama_index.core.program.utils import get_program_for_llm
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.types import PydanticProgramMode
from pydantic import BaseModel, Field

from ogre_kg.utils import quote_cypher

ModelT = TypeVar("ModelT", bound=BaseModel)
DEFAULT_KEYWORD_PROMPT = PromptTemplate(
    "Given a question, generate up to {max_keywords} entity names or graph-relevant "
    "keywords that could match nodes in a knowledge graph.\n"
    "Return only concise candidates.\n"
    "----\n"
    "QUESTION: {question}\n"
    "----"
)
DEFAULT_CHUNK_TERMS_PROMPT = PromptTemplate(
    "Given a question, generate up to {max_terms} concise search phrases for chunk "
    "fulltext search.\n"
    "Focus on domain terms or entities likely to appear verbatim in source text.\n"
    "Return only concise phrases.\n"
    "----\n"
    "QUESTION: {question}\n"
    "----"
)


def _coerce_prompt_template(prompt: str | PromptTemplate) -> PromptTemplate:
    """Normalize prompt inputs to ``PromptTemplate`` instances."""
    if isinstance(prompt, PromptTemplate):
        return prompt
    return PromptTemplate(prompt)


class Keywords(BaseModel):
    """Structured keyword candidates extracted from a user query."""

    names: list[str] = Field(
        default_factory=list,
        description="Possible entity names or graph keywords related to the query.",
    )


class ChunkTerms(BaseModel):
    """Structured chunk-search phrases extracted from a user query."""

    terms: list[str] = Field(
        default_factory=list,
        description="Concise chunk-search phrases likely to appear verbatim in source text.",
    )


class BaseGraphKeywordRetriever(BasePGRetriever):
    """Base retriever that resolves query keywords and fetches graph context.

    Subclasses can implement backend-specific context fetching logic, including
    structured-query backends (Memgraph, Neo4j) or in-memory graph traversal.
    """

    KEYWORD_PROMPT = DEFAULT_KEYWORD_PROMPT

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        llm: LLM | None = None,
        topk_search: int = 3,
        path_depth: int = 1,
        path_limit: int = 30,
        result_limit: int | None = None,
        include_text: bool = True,
        include_properties: bool = False,
        higher_score_is_better: bool = False,
        keyword_prompt: str | PromptTemplate = DEFAULT_KEYWORD_PROMPT,
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
            Maximum number of backend seed nodes to retrieve per keyword.
        path_depth
            Depth passed to ``get_rel_map`` for neighborhood expansion.
        path_limit
            Maximum number of triplets returned by ``get_rel_map``.
        result_limit
            Optional final cap of triplets returned by this retriever.
        include_text
            Whether to inject source chunk text via BasePGRetriever source linkage.
        include_properties
            Whether to include full relation/node properties in triplet text.
        higher_score_is_better
            Score direction from backend seed search.
        keyword_prompt
            Prompt template used to extract graph keywords from the query.
        **kwargs
            Additional retriever options.
        """
        self.llm = llm or Settings.llm
        self.topk = topk_search
        self.path_depth = path_depth
        self.path_limit = path_limit
        self.result_limit = result_limit
        self.higher_score_is_better = higher_score_is_better
        self.keyword_prompt = _coerce_prompt_template(keyword_prompt)
        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    @staticmethod
    def _extract_node_id(record: dict[str, Any]) -> str | None:
        for key in ("node_id", "id", "name"):
            value = record.get(key)
            if value is not None:
                return str(value)
        return None

    def _worst_score(self) -> float:
        return float("-inf") if self.higher_score_is_better else float("inf")

    def _combine_scores(self, left: float, right: float) -> float:
        return max(left, right) if self.higher_score_is_better else min(left, right)

    def _resolve_score(self, source_score: float | None, target_score: float | None) -> float:
        if source_score is None and target_score is None:
            return self._worst_score()
        if source_score is None:
            return float(target_score)
        if target_score is None:
            return float(source_score)
        return self._combine_scores(float(source_score), float(target_score))

    def _score_triplets(
        self,
        triplets: list[Triplet],
        seed_scores: dict[str, float],
    ) -> tuple[list[Triplet], list[float]]:
        dedup_triplets: dict[str, tuple[Triplet, float]] = {}

        for triplet in triplets:
            source_id = triplet[0].id
            target_id = triplet[2].id
            score = self._resolve_score(seed_scores.get(source_id), seed_scores.get(target_id))
            key = f"{source_id}|{triplet[1].id}|{target_id}"

            if key not in dedup_triplets:
                dedup_triplets[key] = (triplet, score)
                continue

            _, existing_score = dedup_triplets[key]
            better_score = self._combine_scores(existing_score, score)
            dedup_triplets[key] = (triplet, better_score)

        sorted_triplets = sorted(
            dedup_triplets.values(),
            key=lambda item: item[1],
            reverse=self.higher_score_is_better,
        )

        if self.result_limit is not None:
            sorted_triplets = sorted_triplets[: self.result_limit]

        return ([item[0] for item in sorted_triplets], [item[1] for item in sorted_triplets])

    @abstractmethod
    def _fetch_keyword_seed_matches(self, keyword: str) -> list[dict[str, Any]]:
        """Fetch seed graph nodes for one keyword."""

    @abstractmethod
    async def _afetch_keyword_seed_matches(self, keyword: str) -> list[dict[str, Any]]:
        """Fetch seed graph nodes for one keyword asynchronously."""

    @staticmethod
    def _normalize_candidates(candidates: list[str], max_items: int) -> list[str]:
        """Normalize structured LLM candidates into a deduplicated list."""
        deduped: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            dedup_key = cleaned.casefold()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            deduped.append(cleaned)
            if len(deduped) >= max_items:
                break

        return deduped

    @staticmethod
    def _should_retry_structured_prediction(error: ValueError) -> bool:
        """Return whether structured prediction should retry in text-parsing mode."""
        return str(error) in {
            "Expected at least one tool call, but got 0 tool calls.",
            "No valid tool calls found.",
        }

    def _structured_predict_with_fallback(
        self,
        output_cls: type[ModelT],
        prompt: PromptTemplate,
        **prompt_args: Any,
    ) -> ModelT:
        """Run structured prediction and retry with text parsing on tool-call failure."""
        try:
            return self.llm.structured_predict(output_cls, prompt, **prompt_args)
        except ValueError as error:
            if not self._should_retry_structured_prediction(error):
                raise

        program = get_program_for_llm(
            output_cls,
            prompt,
            self.llm,
            pydantic_program_mode=PydanticProgramMode.LLM,
        )
        result = program(**prompt_args)
        assert not isinstance(result, list)
        return result

    async def _astructured_predict_with_fallback(
        self,
        output_cls: type[ModelT],
        prompt: PromptTemplate,
        **prompt_args: Any,
    ) -> ModelT:
        """Run async structured prediction and retry with text parsing on tool-call failure."""
        try:
            return await self.llm.astructured_predict(output_cls, prompt, **prompt_args)
        except ValueError as error:
            if not self._should_retry_structured_prediction(error):
                raise

        program = get_program_for_llm(
            output_cls,
            prompt,
            self.llm,
            pydantic_program_mode=PydanticProgramMode.LLM,
        )
        result = await program.acall(**prompt_args)
        assert not isinstance(result, list)
        return result

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract keyword candidates from a question using structured prediction."""
        response = self._structured_predict_with_fallback(
            Keywords,
            self.keyword_prompt,
            question=question,
            max_keywords=self.topk,
        )
        return self._normalize_candidates(response.names, max_items=self.topk)

    async def _aextract_keywords(self, question: str) -> list[str]:
        """Asynchronously extract keyword candidates from a question."""
        response = await self._astructured_predict_with_fallback(
            Keywords,
            self.keyword_prompt,
            question=question,
            max_keywords=self.topk,
        )
        return self._normalize_candidates(response.names, max_items=self.topk)

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve graph-context triplets via keyword seed expansion."""
        keywords = self._extract_keywords(query_bundle.query_str)

        seed_scores: dict[str, float] = {}
        for keyword in keywords:
            for record in self._fetch_keyword_seed_matches(keyword):
                node_id = self._extract_node_id(record)
                if not node_id:
                    continue
                raw_score = record.get("score", self._worst_score())
                score = float(raw_score)
                if node_id in seed_scores:
                    seed_scores[node_id] = self._combine_scores(seed_scores[node_id], score)
                else:
                    seed_scores[node_id] = score

        if not seed_scores:
            return []

        seed_nodes = self._graph_store.get(ids=list(seed_scores.keys()))
        if not seed_nodes:
            return []

        triplets = self._graph_store.get_rel_map(
            seed_nodes,
            depth=self.path_depth,
            limit=self.path_limit,
            ignore_rels=[KG_SOURCE_REL],
        )
        if not triplets:
            return []

        scored_triplets, scores = self._score_triplets(triplets, seed_scores)
        return self._get_nodes_with_score(scored_triplets, scores)

    async def aretrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Asynchronously retrieve graph-context triplets from a user query."""
        keywords = await self._aextract_keywords(query_bundle.query_str)

        seed_scores: dict[str, float] = {}
        for keyword in keywords:
            for record in await self._afetch_keyword_seed_matches(keyword):
                node_id = self._extract_node_id(record)
                if not node_id:
                    continue
                raw_score = record.get("score", self._worst_score())
                score = float(raw_score)
                if node_id in seed_scores:
                    seed_scores[node_id] = self._combine_scores(seed_scores[node_id], score)
                else:
                    seed_scores[node_id] = score

        if not seed_scores:
            return []

        seed_nodes = await self._graph_store.aget(ids=list(seed_scores.keys()))
        if not seed_nodes:
            return []

        triplets = await self._graph_store.aget_rel_map(
            seed_nodes,
            depth=self.path_depth,
            limit=self.path_limit,
            ignore_rels=[KG_SOURCE_REL],
        )
        if not triplets:
            return []

        scored_triplets, scores = self._score_triplets(triplets, seed_scores)
        return self._get_nodes_with_score(scored_triplets, scores)


class GenericKeywordContextRetriever(BaseGraphKeywordRetriever):
    def __init__(
        self,
        graph_store: PropertyGraphStore,
        search_query: str,
        llm: LLM | None = None,
        topk_search: int = 3,
        path_depth: int = 1,
        path_limit: int = 30,
        result_limit: int | None = None,
        include_text: bool = True,
        include_properties: bool = False,
        higher_score_is_better: bool = False,
        keyword_prompt: str | PromptTemplate = DEFAULT_KEYWORD_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            graph_store=graph_store,
            llm=llm,
            topk_search=topk_search,
            path_depth=path_depth,
            path_limit=path_limit,
            result_limit=result_limit,
            include_text=include_text,
            include_properties=include_properties,
            higher_score_is_better=higher_score_is_better,
            keyword_prompt=keyword_prompt,
            **kwargs,
        )
        self.search_query = search_query

    def _build_search_query(self, keyword: str) -> str:
        return self.search_query.format(name=quote_cypher(keyword), topk=self.topk)

    def _fetch_keyword_seed_matches(self, keyword: str) -> list[dict[str, Any]]:
        """Fetch backend seed rows for one keyword."""
        query = self._build_search_query(keyword)
        return self._graph_store.structured_query(query)

    async def _afetch_keyword_seed_matches(self, keyword: str) -> list[dict[str, Any]]:
        """Asynchronously fetch backend seed rows for one keyword."""
        query = self._build_search_query(keyword)
        return await self._graph_store.astructured_query(query)


class GenericChunkKeywordContextRetriever(GenericKeywordContextRetriever):
    """Shared chunk-first retrieval flow for structured-query graph stores."""

    CHUNK_TERMS_PROMPT = DEFAULT_CHUNK_TERMS_PROMPT

    CHUNK_ENTITY_LINK_QUERY = """MATCH (c:Chunk)-[r]->(e:__Entity__)
WHERE c.id IN $chunk_ids AND type(r) IN $allowed_rels
RETURN DISTINCT e.id AS node_id, c.id AS chunk_id"""

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        search_query: str,
        llm: LLM | None = None,
        topk_search: int = 3,
        max_chunk_terms: int = 6,
        chunk_link_rels: tuple[str, ...] = ("MENTIONS",),
        restrict_to_seed_chunks: bool = True,
        keyword_prompt: str | PromptTemplate = DEFAULT_KEYWORD_PROMPT,
        chunk_terms_prompt: str | PromptTemplate = DEFAULT_CHUNK_TERMS_PROMPT,
        **kwargs: Any,
    ) -> None:
        if max_chunk_terms < 1:
            raise ValueError("max_chunk_terms must be >= 1.")
        if not chunk_link_rels or any(not rel for rel in chunk_link_rels):
            raise ValueError("chunk_link_rels must contain at least one non-empty relation name.")

        self.max_chunk_terms = max_chunk_terms
        self.chunk_link_rels = chunk_link_rels
        self.restrict_to_seed_chunks = restrict_to_seed_chunks
        self.chunk_terms_prompt = _coerce_prompt_template(chunk_terms_prompt)

        super().__init__(
            graph_store=graph_store,
            search_query=search_query,
            llm=llm,
            topk_search=topk_search,
            keyword_prompt=keyword_prompt,
            **kwargs,
        )

    @abstractmethod
    def _fetch_chunk_seed_matches(self, term: str) -> list[dict[str, Any]]:
        """Fetch chunk seed rows for one extracted chunk search term."""

    @abstractmethod
    async def _afetch_chunk_seed_matches(self, term: str) -> list[dict[str, Any]]:
        """Asynchronously fetch chunk seed rows for one extracted chunk search term."""

    def _extract_chunk_terms(self, question: str) -> list[str]:
        """Extract structured chunk-search terms from a question."""
        response = self._structured_predict_with_fallback(
            ChunkTerms,
            self.chunk_terms_prompt,
            question=question,
            max_terms=self.max_chunk_terms,
        )
        return self._normalize_candidates(response.terms, max_items=self.max_chunk_terms)

    async def _aextract_chunk_terms(self, question: str) -> list[str]:
        """Asynchronously extract structured chunk-search terms from a question."""
        response = await self._astructured_predict_with_fallback(
            ChunkTerms,
            self.chunk_terms_prompt,
            question=question,
            max_terms=self.max_chunk_terms,
        )
        return self._normalize_candidates(response.terms, max_items=self.max_chunk_terms)

    def _normalize_chunk_terms(self, terms: list[str]) -> list[str]:
        """Normalize extracted chunk terms using the shared candidate policy."""
        return self._normalize_candidates(terms, max_items=self.max_chunk_terms)

    def _collect_seed_scores(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, float]:
        seed_scores: dict[str, float] = {}
        for record in records:
            node_id = self._extract_node_id(record)
            if not node_id:
                continue
            raw_score = record.get("score", self._worst_score())
            score = float(raw_score)
            if node_id in seed_scores:
                seed_scores[node_id] = self._combine_scores(seed_scores[node_id], score)
            else:
                seed_scores[node_id] = score
        return seed_scores

    def _collect_chunk_seed_scores(self, terms: list[str]) -> dict[str, float]:
        seed_scores: dict[str, float] = {}
        for term in terms:
            for record in self._fetch_chunk_seed_matches(term):
                node_id = self._extract_node_id(record)
                if not node_id:
                    continue
                raw_score = record.get("score", self._worst_score())
                score = float(raw_score)
                if node_id in seed_scores:
                    seed_scores[node_id] = self._combine_scores(seed_scores[node_id], score)
                else:
                    seed_scores[node_id] = score
        return seed_scores

    async def _acollect_chunk_seed_scores(self, terms: list[str]) -> dict[str, float]:
        seed_scores: dict[str, float] = {}
        for term in terms:
            for record in await self._afetch_chunk_seed_matches(term):
                node_id = self._extract_node_id(record)
                if not node_id:
                    continue
                raw_score = record.get("score", self._worst_score())
                score = float(raw_score)
                if node_id in seed_scores:
                    seed_scores[node_id] = self._combine_scores(seed_scores[node_id], score)
                else:
                    seed_scores[node_id] = score
        return seed_scores

    def _collect_entity_seed_scores(
        self,
        records: list[dict[str, Any]],
        chunk_scores: dict[str, float],
    ) -> dict[str, float]:
        seed_scores: dict[str, float] = {}
        for record in records:
            node_id = self._extract_node_id(record)
            chunk_id = record.get("chunk_id")
            if not node_id or chunk_id is None:
                continue

            chunk_score = chunk_scores.get(str(chunk_id))
            if chunk_score is None:
                continue

            if node_id in seed_scores:
                seed_scores[node_id] = self._combine_scores(seed_scores[node_id], chunk_score)
            else:
                seed_scores[node_id] = chunk_score
        return seed_scores

    def _fetch_entity_seed_matches(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return self._graph_store.structured_query(
            self.CHUNK_ENTITY_LINK_QUERY,
            param_map={"chunk_ids": chunk_ids, "allowed_rels": list(self.chunk_link_rels)},
        )

    async def _afetch_entity_seed_matches(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return await self._graph_store.astructured_query(
            self.CHUNK_ENTITY_LINK_QUERY,
            param_map={"chunk_ids": chunk_ids, "allowed_rels": list(self.chunk_link_rels)},
        )

    def _filter_triplets_by_chunk_ids(
        self,
        triplets: list[Triplet],
        scores: list[float],
        chunk_ids: set[str],
    ) -> tuple[list[Triplet], list[float]]:
        filtered_triplets: list[Triplet] = []
        filtered_scores: list[float] = []

        for triplet, score in zip(triplets, scores, strict=False):
            source_chunk_id = triplet[0].properties.get("triplet_source_id")
            target_chunk_id = triplet[2].properties.get("triplet_source_id")
            if str(source_chunk_id) in chunk_ids or str(target_chunk_id) in chunk_ids:
                """
                If filtering by seed chunk id is enabled, this ensured that the full
                chunk text will be returned only
                for triplets and nodes originating from the seed chunks:
                (those found by the similarity search).
                """
                if source_chunk_id not in chunk_ids:
                    triplet[0].properties["triplet_source_id"] = None
                if target_chunk_id not in chunk_ids:
                    triplet[2].properties["triplet_source_id"] = None

                filtered_triplets.append(triplet)
                filtered_scores.append(score)

        return filtered_triplets, filtered_scores

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve graph-context triplets from chunk-first keyword search."""
        terms = self._normalize_chunk_terms(self._extract_chunk_terms(query_bundle.query_str))
        if not terms:
            return []

        chunk_seed_scores = self._collect_chunk_seed_scores(terms)
        if not chunk_seed_scores:
            return []

        chunk_ids = list(chunk_seed_scores.keys())
        entity_seed_scores = self._collect_entity_seed_scores(
            self._fetch_entity_seed_matches(chunk_ids),
            chunk_scores=chunk_seed_scores,
        )
        if not entity_seed_scores:
            return []

        seed_nodes = self._graph_store.get(ids=list(entity_seed_scores.keys()))
        if not seed_nodes:
            return []

        triplets = self._graph_store.get_rel_map(
            seed_nodes,
            depth=self.path_depth,
            limit=self.path_limit,
            ignore_rels=[KG_SOURCE_REL],
        )
        if not triplets:
            return []

        scored_triplets, scores = self._score_triplets(triplets, entity_seed_scores)
        if self.restrict_to_seed_chunks:
            scored_triplets, scores = self._filter_triplets_by_chunk_ids(
                scored_triplets,
                scores,
                chunk_ids=set(chunk_seed_scores.keys()),
            )
            if not scored_triplets:
                return []

        return self._get_nodes_with_score(scored_triplets, scores)

    async def aretrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Asynchronously retrieve graph-context triplets from chunk-first keyword search."""
        terms = self._normalize_chunk_terms(
            await self._aextract_chunk_terms(query_bundle.query_str)
        )
        if not terms:
            return []

        chunk_seed_scores = await self._acollect_chunk_seed_scores(terms)
        if not chunk_seed_scores:
            return []

        chunk_ids = list(chunk_seed_scores.keys())
        entity_seed_scores = self._collect_entity_seed_scores(
            await self._afetch_entity_seed_matches(chunk_ids),
            chunk_scores=chunk_seed_scores,
        )
        if not entity_seed_scores:
            return []

        seed_nodes = await self._graph_store.aget(ids=list(entity_seed_scores.keys()))
        if not seed_nodes:
            return []

        triplets = await self._graph_store.aget_rel_map(
            seed_nodes,
            depth=self.path_depth,
            limit=self.path_limit,
            ignore_rels=[KG_SOURCE_REL],
        )
        if not triplets:
            return []

        scored_triplets, scores = self._score_triplets(triplets, entity_seed_scores)
        if self.restrict_to_seed_chunks:
            scored_triplets, scores = self._filter_triplets_by_chunk_ids(
                scored_triplets,
                scores,
                chunk_ids=set(chunk_seed_scores.keys()),
            )
            if not scored_triplets:
                return []

        return self._get_nodes_with_score(scored_triplets, scores)
