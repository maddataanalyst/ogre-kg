"""Entity processing exports for OGRE KG."""

from ogre_kg.kg_processors.entity_processing.entity_disambiguation import (
    EntityDisambiguationProcessor,
)
from ogre_kg.kg_processors.entity_processing.entity_merger import (
    CanonicalEntitySelectionStrategy,
    EntityMerger,
    FalkorDBEntityMerger,
    FalkorDBSynonymCreator,
    MemgraphEntityMerger,
    MemgraphSynonymCreator,
    Neo4jEntityMerger,
    Neo4jSynonymCreator,
    SynonymCreator,
)
from ogre_kg.kg_processors.entity_processing.entity_similarity_finders import (
    CustomEmbeddingsSimilarityFinder,
    EntitySimilarityFinder,
    ExactMatchEntitySimilarityFinder,
    FalkorDBVectorEntitySimilarityFinder,
    FuzzyEntitySimilarityFinder,
    MemgraphCypherEntitySimilarityFinder,
    Neo4jGDSEntitySimilarityFinder,
)

__all__ = [
    "CanonicalEntitySelectionStrategy",
    "CustomEmbeddingsSimilarityFinder",
    "EntityDisambiguationProcessor",
    "EntityMerger",
    "EntitySimilarityFinder",
    "ExactMatchEntitySimilarityFinder",
    "FalkorDBEntityMerger",
    "FalkorDBSynonymCreator",
    "FalkorDBVectorEntitySimilarityFinder",
    "FuzzyEntitySimilarityFinder",
    "MemgraphCypherEntitySimilarityFinder",
    "MemgraphEntityMerger",
    "MemgraphSynonymCreator",
    "Neo4jEntityMerger",
    "Neo4jGDSEntitySimilarityFinder",
    "Neo4jSynonymCreator",
    "SynonymCreator",
]
