"""Entity disambiguation processor composing similarity finders and mergers."""

from __future__ import annotations

from typing import Any

from ogre_kg.kg_processors.entity_processing.entity_merger import EntityMerger
from ogre_kg.kg_processors.entity_processing.entity_similarity_finders import (
    EntitySimilarityFinder,
)


class EntityDisambiguationProcessor:
    """Composable entity disambiguation processor.

    Combines any ``EntitySimilarityFinder`` with any ``EntityMerger``.
    The ``process()`` method runs the full find-then-process pipeline.

    This design allows mixing any finder with any merger, for example:
    - ``FuzzyEntitySimilarityFinder`` + ``Neo4jEntityMerger``
    - ``MemgraphCypherEntitySimilarityFinder`` + ``MemgraphSynonymCreator``
    - ``Neo4jGDSEntitySimilarityFinder`` + ``Neo4jSynonymCreator``

    Each backend-specific finder and merger validates its own graph store
    at init time, so invalid combinations fail early.

    Parameters
    ----------
    similarity_finder
        Component responsible for finding groups of similar entities.
    merger
        Component responsible for applying the chosen group-processing strategy
        in the graph store.
    """

    def __init__(
        self,
        similarity_finder: EntitySimilarityFinder,
        merger: EntityMerger,
    ) -> None:
        self.similarity_finder = similarity_finder
        self.merger = merger

    def process(self) -> list[dict[str, Any]]:
        """Run end-to-end disambiguation: find similar entities, then process.

        Returns
        -------
        list[dict[str, Any]]
            Backend results from the configured merger.
        """
        entity_groups = self.similarity_finder.find_similar_entities()
        return self.merger.merge_entities(entity_groups)
