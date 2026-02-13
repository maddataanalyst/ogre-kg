"""Entity similarity finder implementations for property graph backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from llama_index.core.graph_stores.types import EntityNode, PropertyGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rapidfuzz import fuzz
from tqdm.auto import tqdm

from ogre_kg.utils import (
    GraphStoreBackend,
    PropertyGraphStoreProvider,
    StructuredQueryCapableStore,
    build_entity_groups,
    detect_graph_store_backend,
    resolve_graph_store,
)


class EntitySimilarityFinder(ABC):
    """Base class for entity similarity finders.

    Parameters
    ----------
    source
        Graph store or index-like object exposing ``property_graph_store``.
    similarity_threshold
        Minimum similarity score for considering two entities as similar.
    """

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.graph_store = resolve_graph_store(source)
        self.similarity_threshold = similarity_threshold

    @abstractmethod
    def find_similar_entities(self) -> list[set[str]]:
        """Find groups of similar entities that should be merged.

        Returns
        -------
        list[set[str]]
            Groups of entity names considered similar.
        """


class MemgraphCypherEntitySimilarityFinder(EntitySimilarityFinder):
    """Entity similarity finder using Memgraph Cypher cosine similarity.

    Uses Memgraph's ``node_similarity.cosine`` procedure on pre-computed
    embeddings to find similar entity pairs, then groups them into
    connected components via union-find.

    Validates at init time that the provided graph store is a Memgraph backend.

    Parameters
    ----------
    source
        Memgraph graph store or index-like provider.
    embedding_attr
        Node property name containing embedding vectors.
    similarity_threshold
        Minimum cosine similarity for a pair to be considered similar.
    """

    SIMILARITY_QUERY = """MATCH pth=(n: __Entity__)-[r]->(n2)
WITH PROJECT(pth) AS subg
CALL node_similarity.cosine(subg, "{embedding_attr}")
YIELD node1, node2, similarity
WITH node1, node2, similarity
WHERE similarity > {threshold} AND node1:__Entity__ AND node2:__Entity__ AND node1<>node2
RETURN node1.name AS name1, node2.name AS name2, similarity
ORDER BY similarity DESC"""

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        embedding_attr: str,
        similarity_threshold: float = 0.5,
    ) -> None:
        super().__init__(source, similarity_threshold)
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.MEMGRAPH:
            raise ValueError(
                f"MemgraphCypherEntitySimilarityFinder requires a Memgraph graph store, "
                f"but detected backend: {backend.value}"
            )
        self.embedding_attr = embedding_attr

    def find_similar_entities(self) -> list[set[str]]:
        """Find similar entity groups using Memgraph cosine similarity.

        Returns
        -------
        list[set[str]]
            Connected components of similar entities.
        """
        query = self.SIMILARITY_QUERY.format(
            embedding_attr=self.embedding_attr,
            threshold=self.similarity_threshold,
        )
        pairs = self.graph_store.structured_query(query)
        return build_entity_groups(pairs)


class Neo4jGDSEntitySimilarityFinder(EntitySimilarityFinder):
    """Entity similarity finder using Neo4j Graph Data Science node similarity.

    Uses ``gds.nodeSimilarity.stream`` on a pre-projected GDS graph to find
    similar entity pairs, then groups them into connected components via
    union-find.

    Validates at init time that the provided graph store is a Neo4j backend.

    Parameters
    ----------
    source
        Neo4j graph store or index-like provider.
    graph_name
        Name of the pre-projected GDS graph to run similarity on.
    similarity_threshold
        Minimum similarity cutoff passed to ``similarityCutoff``.
    top_k
        Maximum number of similar neighbors per node.
    similarity_metric
        Similarity metric to use: ``JACCARD``, ``OVERLAP``, or ``COSINE``.
    """

    SIMILARITY_QUERY = (
        "MATCH (a:`__Entity__`), (b:`__Entity__`)\n"
        "WHERE id(a) < id(b)\n"
        "WITH a.name as name1, b.name as name2,\n"
        "  vector.similarity.{similarity_metric}(a.embedding, b.embedding) AS sim\n"
        "WHERE sim > {threshold}\n"
        "RETURN name1, name2, sim\n"
        "ORDER BY sim DESC"
    )

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider,
        embedding_attr: str,
        similarity_threshold: float = 0.5,
        similarity_metric: Literal["cosine", "euclidean"] = "cosine",
    ) -> None:
        super().__init__(source, similarity_threshold)
        backend = detect_graph_store_backend(self.graph_store)
        if backend != GraphStoreBackend.NEO4J:
            raise ValueError(
                f"Neo4jGDSEntitySimilarityFinder requires a Neo4j graph store, "
                f"but detected backend: {backend.value}"
            )
        self.embedding_attr = embedding_attr
        self.similarity_metric = similarity_metric

    def find_similar_entities(self) -> list[set[str]]:
        """Find similar entity groups using Neo4j GDS node similarity.

        Returns
        -------
        list[set[str]]
            Connected components of similar entities.
        """
        query = self.SIMILARITY_QUERY.format(
            threshold=self.similarity_threshold,
            similarity_metric=self.similarity_metric,
        )
        pairs = self.graph_store.structured_query(query)
        return build_entity_groups(pairs)


class FuzzyEntitySimilarityFinder(EntitySimilarityFinder):
    """Entity similarity finder using fuzzy string matching (rapidfuzz).

    Compares entity names pairwise using token-sort ratio from rapidfuzz.
    Works with any ``PropertyGraphStore`` that supports ``get()``.
    This is a store-independent finder (works with Memgraph, Neo4j, in-memory).

    Parameters
    ----------
    source
        Graph store or index-like object. Must support ``get()`` returning
        entity nodes.
    similarity_threshold
        Minimum fuzzy ratio (0-100) for considering two entities similar.
    """

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider | PropertyGraphStore,
        similarity_threshold: float = 80.0,
    ) -> None:
        self.graph_store = _resolve_graph_store_with_get(source)
        self.similarity_threshold = similarity_threshold

    def find_similar_entities(self) -> list[set[str]]:
        """Find similar entity groups using fuzzy string matching.

        Returns
        -------
        list[set[str]]
            Groups of entity names with high fuzzy similarity.
        """
        all_entity_nodes = [node for node in self.graph_store.get() if isinstance(node, EntityNode)]
        fuzzy_matched_entities: dict[str, set[str]] = defaultdict(set)
        fuzzy_already_matched_entities: set[str] = set()

        for idx, entity_1 in enumerate(all_entity_nodes):
            if entity_1.name in fuzzy_already_matched_entities:
                continue

            for jdx in range(idx + 1, len(all_entity_nodes)):
                entity_2 = all_entity_nodes[jdx]
                if entity_2.name in fuzzy_already_matched_entities:
                    continue

                if fuzz.ratio(entity_1.name, entity_2.name) > self.similarity_threshold:
                    fuzzy_matched_entities[entity_1.name].add(entity_1.name)
                    fuzzy_matched_entities[entity_1.name].add(entity_2.name)
                    fuzzy_already_matched_entities.add(entity_2.name)

        return list(fuzzy_matched_entities.values())


class CustomEmbeddingsSimilarityFinder(EntitySimilarityFinder):
    """Entity similarity finder using custom HuggingFace embeddings.

    Computes embeddings on the fly for entity names and compares them
    via cosine similarity. Works with any ``PropertyGraphStore`` that
    supports ``get()``.
    This is a store-independent finder (works with Memgraph, Neo4j, in-memory).

    Parameters
    ----------
    source
        Graph store or index-like object. Must support ``get()`` returning
        entity nodes.
    hugging_face_model_name
        HuggingFace model identifier for computing text embeddings.
    similarity_threshold
        Minimum cosine similarity for considering two entities similar.
    """

    def __init__(
        self,
        source: StructuredQueryCapableStore | PropertyGraphStoreProvider | PropertyGraphStore,
        hugging_face_model_name: str,
        similarity_threshold: float = 0.5,
    ) -> None:
        self.graph_store = _resolve_graph_store_with_get(source)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = HuggingFaceEmbedding(model_name=hugging_face_model_name)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Parameters
        ----------
        v1, v2
            Input vectors.

        Returns
        -------
        float
            Cosine similarity in [0, 1]. Returns 0 if either vector has zero norm.
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if not norm1 or not norm2:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def get_cached_embedding(
        name: str,
        embeddings_cache: dict[str, np.ndarray],
        embed_model: HuggingFaceEmbedding,
    ) -> np.ndarray:
        """Retrieve or compute and cache an embedding for a given name.

        Parameters
        ----------
        name
            Entity name to embed.
        embeddings_cache
            Cache dict mapping names to precomputed embeddings.
        embed_model
            HuggingFace embedding model.

        Returns
        -------
        np.ndarray
            Embedding vector for the name.
        """
        if name in embeddings_cache:
            return embeddings_cache[name]
        embedding = np.array(embed_model.get_text_embedding(name))
        embeddings_cache[name] = embedding
        return embedding

    def find_similar_entities(self) -> list[set[str]]:
        """Find similar entity groups using embedding cosine similarity.

        Returns
        -------
        list[set[str]]
            Groups of entity names with high embedding similarity.
        """
        embedding_matched_entities: dict[str, set[str]] = defaultdict(set)
        embedding_already_matched_entities: set[str] = set()

        embeddings_cache: dict[str, np.ndarray] = {}
        all_entity_nodes = [node for node in self.graph_store.get() if isinstance(node, EntityNode)]

        for idx, entity_1 in tqdm(enumerate(all_entity_nodes)):
            if entity_1.name in embedding_already_matched_entities:
                continue

            name1_embedding = self.get_cached_embedding(
                entity_1.name, embeddings_cache, self.embedding_model
            )

            for jdx in range(idx + 1, len(all_entity_nodes)):
                entity_2 = all_entity_nodes[jdx]
                if entity_2.name in embedding_already_matched_entities:
                    continue

                name2_embedding = self.get_cached_embedding(
                    entity_2.name, embeddings_cache, self.embedding_model
                )

                cosine = self.cosine_similarity(name1_embedding, name2_embedding)
                if cosine > self.similarity_threshold:
                    embedding_matched_entities[entity_1.name].add(entity_1.name)
                    embedding_matched_entities[entity_1.name].add(entity_2.name)
                    embedding_already_matched_entities.add(entity_2.name)

        return list(embedding_matched_entities.values())


def _resolve_graph_store_with_get(source: Any) -> Any:
    """Resolve a graph store ensuring it supports the ``get()`` method.

    Parameters
    ----------
    source
        Graph store, index-like provider, or ``PropertyGraphStore``.

    Returns
    -------
    Any
        Resolved graph store with ``get()`` method.

    Raises
    ------
    ValueError
        If the resolved store does not support ``get()``.
    """
    # Direct PropertyGraphStore with get()
    if hasattr(source, "get"):
        return source

    # Index-like object wrapping a store
    graph_store = getattr(source, "property_graph_store", None)
    if graph_store is not None and hasattr(graph_store, "get"):
        return graph_store

    raise ValueError(
        "FuzzyEntitySimilarityFinder and CustomEmbeddingsSimilarityFinder require "
        "a graph store with a get() method (e.g. PropertyGraphStore)."
    )
