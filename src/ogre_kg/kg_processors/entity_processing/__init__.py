"""Entity processing exports for OGRE KG."""

from ogre_kg.kg_processors.entity_processing.entity_disambiguation import (
    EntityDisambiguationProcessor,
)
from ogre_kg.kg_processors.entity_processing.entity_merger import (
    EntityMerger,
    MemgraphEntityMerger,
    MemgraphSynonymCreator,
    Neo4jEntityMerger,
    Neo4jSynonymCreator,
    SynonymCreator,
)
from ogre_kg.kg_processors.entity_processing.entity_similarity_finders import (
    CustomEmbeddingsSimilarityFinder,
    EntitySimilarityFinder,
    FuzzyEntitySimilarityFinder,
    MemgraphCypherEntitySimilarityFinder,
    Neo4jGDSEntitySimilarityFinder,
)

__all__ = [
    "CustomEmbeddingsSimilarityFinder",
    "EntityDisambiguationProcessor",
    "EntityMerger",
    "EntitySimilarityFinder",
    "FuzzyEntitySimilarityFinder",
    "MemgraphCypherEntitySimilarityFinder",
    "MemgraphEntityMerger",
    "MemgraphSynonymCreator",
    "Neo4jEntityMerger",
    "Neo4jGDSEntitySimilarityFinder",
    "Neo4jSynonymCreator",
    "SynonymCreator",
]
