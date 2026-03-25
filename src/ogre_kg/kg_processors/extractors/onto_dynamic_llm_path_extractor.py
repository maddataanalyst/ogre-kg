"""Ontology-aware path extractors."""

from __future__ import annotations

from collections.abc import Callable

from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.indices.property_graph.transformations.dynamic_llm import (
    DEFAULT_DYNAMIC_EXTRACT_PROMPT,
    DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import PromptTemplate

DEFAULT_ONTOLOGY_RELATION_TYPES = [
    "SUBCLASS_OF",
    "EQUIVALENT_OF",
    "COMPLEMENT_OF",
    "PART_OF",
]
DEFAULT_ONTO_DYNAMIC_EXTRACT_PROMPT = DEFAULT_DYNAMIC_EXTRACT_PROMPT
DEFAULT_ONTO_DYNAMIC_EXTRACT_PROPS_PROMPT = DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT


def _merge_relation_types(
    allowed_relation_types: list[str] | None,
) -> list[str]:
    """Merge user-specified relation types with ontology defaults.

    Parameters
    ----------
    allowed_relation_types
        Optional user-provided relation types.

    Returns
    -------
    list[str]
        Deduplicated relation types preserving the user-provided order first,
        followed by missing ontology defaults.
    """
    merged_relation_types = list(dict.fromkeys(allowed_relation_types or []))

    for relation_type in DEFAULT_ONTOLOGY_RELATION_TYPES:
        if relation_type not in merged_relation_types:
            merged_relation_types.append(relation_type)

    return merged_relation_types


def _resolve_extract_prompt(
    extract_prompt: str | PromptTemplate | None,
    allowed_entity_props: list[str] | list[tuple[str, str]] | None,
    allowed_relation_props: list[str] | list[tuple[str, str]] | None,
) -> PromptTemplate:
    """Resolve the effective extractor prompt template.

    Parameters
    ----------
    extract_prompt
        User-specified prompt override.
    allowed_entity_props
        Optional entity properties passed to the extractor.
    allowed_relation_props
        Optional relation properties passed to the extractor.

    Returns
    -------
    PromptTemplate
        Concrete prompt template used by the parent extractor.
    """
    if isinstance(extract_prompt, PromptTemplate):
        return extract_prompt
    if isinstance(extract_prompt, str):
        return PromptTemplate(extract_prompt)
    if allowed_entity_props is not None or allowed_relation_props is not None:
        return DEFAULT_ONTO_DYNAMIC_EXTRACT_PROPS_PROMPT
    return DEFAULT_ONTO_DYNAMIC_EXTRACT_PROMPT


class OntoDynamicLLMPathExtractor(DynamicLLMPathExtractor):
    """Dynamic path extractor seeded with ontology-oriented relation defaults.

    Parameters
    ----------
    llm
        Language model used for extraction.
    extract_prompt
        Prompt template used by the parent extractor.
    parse_fn
        Optional parser for transforming LLM output into graph triplets.
    max_triplets_per_chunk
        Maximum number of triplets extracted from a single chunk.
    num_workers
        Number of parallel workers used during extraction.
    allowed_entity_types
        Optional initial ontology entity types.
    allowed_entity_props
        Optional entity property names or ``(name, description)`` pairs.
    allowed_relation_types
        Optional relation types. When provided, ontology defaults are appended
        unless already present.
    allowed_relation_props
        Optional relation property names or ``(name, description)`` pairs.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        extract_prompt: str | PromptTemplate | None = None,
        parse_fn: Callable | None = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
        allowed_entity_types: list[str] | None = None,
        allowed_entity_props: list[str] | list[tuple[str, str]] | None = None,
        allowed_relation_types: list[str] | None = None,
        allowed_relation_props: list[str] | list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            extract_prompt=_resolve_extract_prompt(
                extract_prompt=extract_prompt,
                allowed_entity_props=allowed_entity_props,
                allowed_relation_props=allowed_relation_props,
            ),
            parse_fn=parse_fn,
            max_triplets_per_chunk=max_triplets_per_chunk,
            num_workers=num_workers,
            allowed_entity_types=allowed_entity_types,
            allowed_entity_props=allowed_entity_props,
            allowed_relation_types=_merge_relation_types(allowed_relation_types),
            allowed_relation_props=allowed_relation_props,
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "OntoDynamicLLMPathExtractor"
