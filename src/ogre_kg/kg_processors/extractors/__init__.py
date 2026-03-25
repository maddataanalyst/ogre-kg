"""Extractor exports for OGRE KG."""

from ogre_kg.kg_processors.extractors.onto_dynamic_llm_path_extractor import (
    DEFAULT_ONTO_DYNAMIC_EXTRACT_PROMPT,
    DEFAULT_ONTO_DYNAMIC_EXTRACT_PROPS_PROMPT,
    DEFAULT_ONTOLOGY_RELATION_TYPES,
    OntoDynamicLLMPathExtractor,
)
from ogre_kg.kg_processors.extractors.tabular_path_extractor import (
    ForeignKeySpec,
    RelationColumnSpec,
    TabularPathExtractor,
    TabularTableSpec,
)

__all__ = [
    "DEFAULT_ONTO_DYNAMIC_EXTRACT_PROMPT",
    "DEFAULT_ONTO_DYNAMIC_EXTRACT_PROPS_PROMPT",
    "DEFAULT_ONTOLOGY_RELATION_TYPES",
    "ForeignKeySpec",
    "OntoDynamicLLMPathExtractor",
    "RelationColumnSpec",
    "TabularPathExtractor",
    "TabularTableSpec",
]
