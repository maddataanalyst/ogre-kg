"""Tests for OGRE KG extractors."""

from __future__ import annotations

import inspect
from io import StringIO

import pandas as pd
import pytest
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.llms.mock import MockLLM
from llama_index.core.prompts import PromptTemplate

from ogre_kg.kg_processors import (
    DEFAULT_ONTO_DYNAMIC_EXTRACT_PROMPT,
    DEFAULT_ONTO_DYNAMIC_EXTRACT_PROPS_PROMPT,
    ForeignKeySpec,
    OntoDynamicLLMPathExtractor,
    RelationColumnSpec,
    TabularPathExtractor,
    TabularTableSpec,
)
from ogre_kg.kg_processors.extractors import DEFAULT_ONTOLOGY_RELATION_TYPES


def _entity_by_name(entity_nodes: list[EntityNode], name: str) -> EntityNode:
    """Return the entity node matching a given name."""
    return next(entity_node for entity_node in entity_nodes if entity_node.name == name)


def _csv_stream(content: str) -> StringIO:
    """Return an in-memory CSV text stream."""
    return StringIO(content)


class TestOntoDynamicLLMPathExtractor:
    def test_constructor_signature_matches_parent(self):
        # given
        parent_parameters = inspect.signature(DynamicLLMPathExtractor.__init__).parameters

        # when
        child_parameters = inspect.signature(OntoDynamicLLMPathExtractor.__init__).parameters

        # then
        assert tuple(child_parameters) == tuple(parent_parameters)
        for parameter_name in parent_parameters:
            assert (
                child_parameters[parameter_name].default
                == parent_parameters[parameter_name].default
            )

    def test_uses_ontology_defaults_when_no_relations_provided(self):
        # given
        llm = MockLLM()

        # when
        extractor = OntoDynamicLLMPathExtractor(llm=llm)

        # then
        assert extractor.allowed_relation_types == DEFAULT_ONTOLOGY_RELATION_TYPES

    def test_resolves_default_extract_prompt_when_not_provided(self):
        # given
        llm = MockLLM()

        # when
        extractor = OntoDynamicLLMPathExtractor(llm=llm)

        # then
        assert extractor.extract_prompt is DEFAULT_ONTO_DYNAMIC_EXTRACT_PROMPT

    def test_resolves_property_aware_default_prompt_when_props_are_enabled(self):
        # given
        llm = MockLLM()

        # when
        extractor = OntoDynamicLLMPathExtractor(
            llm=llm,
            allowed_entity_props=["definition"],
        )

        # then
        assert extractor.extract_prompt is DEFAULT_ONTO_DYNAMIC_EXTRACT_PROPS_PROMPT

    def test_coerces_string_extract_prompt_override(self):
        # given
        llm = MockLLM()
        prompt_text = "Wyodrebnij relacje ontologiczne.\nQUESTION: {text}"

        # when
        extractor = OntoDynamicLLMPathExtractor(
            llm=llm,
            extract_prompt=prompt_text,
        )

        # then
        assert isinstance(extractor.extract_prompt, PromptTemplate)
        assert extractor.extract_prompt.template == prompt_text

    def test_preserves_prompt_template_extract_prompt_override(self):
        # given
        llm = MockLLM()
        prompt = PromptTemplate("Extract ontology facts.\nTEXT: {text}")

        # when
        extractor = OntoDynamicLLMPathExtractor(
            llm=llm,
            extract_prompt=prompt,
        )

        # then
        assert extractor.extract_prompt is prompt

    def test_appends_ontology_defaults_to_user_relations(self):
        # given
        llm = MockLLM()

        # when
        extractor = OntoDynamicLLMPathExtractor(
            llm=llm,
            allowed_relation_types=["CAUSES", "PART_OF"],
        )

        # then
        assert extractor.allowed_relation_types == [
            "CAUSES",
            "PART_OF",
            "SUBCLASS_OF",
            "EQUIVALENT_OF",
            "COMPLEMENT_OF",
        ]

    def test_deduplicates_relations_while_preserving_order(self):
        # given
        llm = MockLLM()

        # when
        extractor = OntoDynamicLLMPathExtractor(
            llm=llm,
            allowed_relation_types=["EQUIVALENT_OF", "CAUSES", "PART_OF", "CAUSES"],
        )

        # then
        assert extractor.allowed_relation_types == [
            "EQUIVALENT_OF",
            "CAUSES",
            "PART_OF",
            "SUBCLASS_OF",
            "COMPLEMENT_OF",
        ]

    def test_passes_other_constructor_arguments_through(self):
        # given
        llm = MockLLM()
        relation_props = [("confidence", "Confidence score")]

        # when
        extractor = OntoDynamicLLMPathExtractor(
            llm=llm,
            max_triplets_per_chunk=7,
            num_workers=2,
            allowed_entity_types=["Class"],
            allowed_entity_props=["definition"],
            allowed_relation_props=relation_props,
        )

        # then
        assert extractor.llm is llm
        assert extractor.max_triplets_per_chunk == 7
        assert extractor.num_workers == 2
        assert extractor.allowed_entity_types == ["Class"]
        assert extractor.allowed_entity_props == ["definition"]
        assert extractor.allowed_relation_props == [
            "Property `confidence` with description (Confidence score)"
        ]


class TestTabularPathExtractor:
    def test_load_documents_returns_one_document_per_row_from_stream(self):
        # given
        companies_stream = _csv_stream(
            "company_id,industry,revenue_usd\nMSFT,IT,123456789\nAAPL,Consumer Electronics,42\n"
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    stream=companies_stream,
                )
            ]
        )

        # when
        documents = extractor.load_documents()

        # then
        assert len(documents) == 2
        assert documents[0].metadata["ogre_kg_table_name"] == "companies"
        assert documents[0].metadata["ogre_kg_row_data"]["company_id"] == "MSFT"

    def test_single_table_creates_main_entity_relation_and_numeric_property_from_dataframe(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": "MSFT", "industry": "IT", "revenue_usd": 123456789},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]

        main_entity = _entity_by_name(entity_nodes, "companies:company_id=MSFT")
        industry_entity = _entity_by_name(entity_nodes, "Industry:IT")
        assert main_entity.label == "Companies"
        assert main_entity.properties["company_id"] == "MSFT"
        assert main_entity.properties["revenue_usd"] == 123456789
        assert industry_entity.label == "Industry"
        assert relations[0].label == "HAS_INDUSTRY"
        assert relations[0].source_id == main_entity.name
        assert relations[0].target_id == industry_entity.name

    def test_multi_table_explicit_foreign_key_creates_link_to_target_row_from_dataframes(self):
        # given
        departments_dataframe = pd.DataFrame(
            [
                {"dept_id": "D10", "name": "Engineering"},
            ]
        )
        employees_dataframe = pd.DataFrame(
            [
                {"emp_id": "E1", "dept_id": "D10", "title": "Engineer"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="departments",
                    primary_key="dept_id",
                    dataframe=departments_dataframe,
                ),
                TabularTableSpec(
                    name="employees",
                    primary_key="emp_id",
                    dataframe=employees_dataframe,
                    foreign_keys=[
                        ForeignKeySpec(
                            source_columns="dept_id",
                            target_table="departments",
                            target_columns="dept_id",
                            relation_label="BELONGS_TO_DEPARTMENT",
                        )
                    ],
                ),
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        employee_node = next(
            node for node in transformed_nodes if node.metadata["ogre_kg_table_name"] == "employees"
        )
        entity_nodes = employee_node.metadata["nodes"]
        relations = employee_node.metadata["relations"]

        employee_entity = _entity_by_name(entity_nodes, "employees:emp_id=E1")
        department_entity = _entity_by_name(entity_nodes, "departments:dept_id=D10")
        title_entity = _entity_by_name(entity_nodes, "Title:Engineer")

        assert employee_entity.label == "Employees"
        assert department_entity.label == "Departments"
        assert title_entity.label == "Title"
        assert [relation.label for relation in relations] == [
            "BELONGS_TO_DEPARTMENT",
            "HAS_TITLE",
        ]
        assert relations[0].target_id == department_entity.name

    def test_composite_primary_key_and_foreign_key_resolution(self):
        # given
        warehouses_dataframe = pd.DataFrame(
            [
                {"country_code": "PL", "warehouse_id": "W1", "name": "Warsaw"},
            ]
        )
        inventory_dataframe = pd.DataFrame(
            [
                {
                    "inventory_id": "I1",
                    "country_code": "PL",
                    "warehouse_id": "W1",
                    "sku": "SKU-1",
                },
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="warehouses",
                    primary_key=["country_code", "warehouse_id"],
                    dataframe=warehouses_dataframe,
                ),
                TabularTableSpec(
                    name="inventory",
                    primary_key="inventory_id",
                    dataframe=inventory_dataframe,
                    foreign_keys=[
                        ForeignKeySpec(
                            source_columns=["country_code", "warehouse_id"],
                            target_table="warehouses",
                            target_columns=["country_code", "warehouse_id"],
                            relation_label="STORED_IN",
                        )
                    ],
                ),
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        inventory_node = next(
            node for node in transformed_nodes if node.metadata["ogre_kg_table_name"] == "inventory"
        )
        entity_nodes = inventory_node.metadata["nodes"]
        relations = inventory_node.metadata["relations"]

        warehouse_entity = _entity_by_name(
            entity_nodes,
            "warehouses:country_code=PL|warehouse_id=W1",
        )
        assert warehouse_entity.properties["country_code"] == "PL"
        assert warehouse_entity.properties["warehouse_id"] == "W1"
        assert relations[0].label == "STORED_IN"
        assert relations[0].target_id == warehouse_entity.name

    def test_property_columns_override_text_relation_inference(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": "MSFT", "revenue_text": "123 USD"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                    property_columns=["revenue_text"],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]
        main_entity = _entity_by_name(entity_nodes, "companies:company_id=MSFT")

        assert main_entity.properties["revenue_text"] == "123 USD"
        assert relations == []
        assert len(entity_nodes) == 1

    def test_relation_column_spec_overrides_default_relation_and_target_label(self):
        # given
        employees_dataframe = pd.DataFrame(
            [
                {"emp_id": "E1", "title": "Engineer"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="employees",
                    primary_key="emp_id",
                    dataframe=employees_dataframe,
                    relation_columns=[
                        RelationColumnSpec(
                            column="title",
                            relation_label="WORKS_AS",
                            target_label="JobTitle",
                        )
                    ],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]
        title_entity = _entity_by_name(entity_nodes, "JobTitle:Engineer")

        assert title_entity.label == "JobTitle"
        assert relations[0].label == "WORKS_AS"

    def test_list_valued_relation_column_creates_multiple_entities_and_edges(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": 1, "industry": "Tech", "departments": "R&D, Sales"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                    relation_columns=[
                        RelationColumnSpec(column="departments", split_separator=",")
                    ],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]

        _entity_by_name(entity_nodes, "Departments:R&D")
        _entity_by_name(entity_nodes, "Departments:Sales")
        assert [relation.label for relation in relations] == [
            "HAS_INDUSTRY",
            "HAS_DEPARTMENTS",
            "HAS_DEPARTMENTS",
        ]
        assert [relation.target_id for relation in relations[1:]] == [
            "Departments:R&D",
            "Departments:Sales",
        ]

    def test_list_valued_relation_column_skips_empty_tokens_and_deduplicates(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": 1, "departments": "R&D, , Sales, R&D "},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                    relation_columns=[
                        RelationColumnSpec(column="departments", split_separator=",")
                    ],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]

        assert len(relations) == 2
        assert len(entity_nodes) == 3
        assert [relation.target_id for relation in relations] == [
            "Departments:R&D",
            "Departments:Sales",
        ]

    def test_list_valued_relation_column_preserves_whitespace_when_disabled(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": 1, "departments": "R&D, Sales "},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                    relation_columns=[
                        RelationColumnSpec(
                            column="departments",
                            split_separator=",",
                            strip_values=False,
                        )
                    ],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        relations = transformed_nodes[0].metadata["relations"]
        assert [relation.target_id for relation in relations] == [
            "Departments:R&D",
            "Departments: Sales ",
        ]

    def test_list_valued_relation_column_respects_custom_labels(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": 1, "departments": "R&D,Sales"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                    relation_columns=[
                        RelationColumnSpec(
                            column="departments",
                            relation_label="HAS_DEPARTMENT",
                            target_label="Department",
                            split_separator=",",
                        )
                    ],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]

        _entity_by_name(entity_nodes, "Department:R&D")
        _entity_by_name(entity_nodes, "Department:Sales")
        assert [relation.label for relation in relations] == [
            "HAS_DEPARTMENT",
            "HAS_DEPARTMENT",
        ]

    def test_relation_column_spec_takes_precedence_over_scalar_inference(self):
        # given
        companies_dataframe = pd.DataFrame(
            [
                {"company_id": 1, "codes": 101},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="companies",
                    primary_key="company_id",
                    dataframe=companies_dataframe,
                    relation_columns=[RelationColumnSpec(column="codes")],
                )
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        entity_nodes = transformed_nodes[0].metadata["nodes"]
        relations = transformed_nodes[0].metadata["relations"]
        main_entity = _entity_by_name(entity_nodes, "companies:company_id=1")

        _entity_by_name(entity_nodes, "Codes:101")
        assert "codes" not in main_entity.properties
        assert [relation.label for relation in relations] == ["HAS_CODES"]

    def test_stream_and_dataframe_sources_can_be_mixed(self):
        # given
        departments_stream = _csv_stream("dept_id,name\nD10,Engineering\n")
        employees_dataframe = pd.DataFrame(
            [
                {"emp_id": "E1", "dept_id": "D10"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="departments",
                    primary_key="dept_id",
                    stream=departments_stream,
                ),
                TabularTableSpec(
                    name="employees",
                    primary_key="emp_id",
                    dataframe=employees_dataframe,
                    foreign_keys=[
                        ForeignKeySpec(
                            source_columns="dept_id",
                            target_table="departments",
                            target_columns="dept_id",
                            relation_label="BELONGS_TO_DEPARTMENT",
                        )
                    ],
                ),
            ]
        )
        documents = extractor.load_documents()

        # when
        transformed_nodes = extractor(documents)

        # then
        employee_node = next(
            node for node in transformed_nodes if node.metadata["ogre_kg_table_name"] == "employees"
        )
        relations = employee_node.metadata["relations"]
        assert [relation.label for relation in relations] == ["BELONGS_TO_DEPARTMENT"]

    def test_unresolved_foreign_key_raises_error(self):
        # given
        departments_dataframe = pd.DataFrame(
            [
                {"dept_id": "D10", "name": "Engineering"},
            ]
        )
        employees_dataframe = pd.DataFrame(
            [
                {"emp_id": "E1", "dept_id": "D99"},
            ]
        )
        extractor = TabularPathExtractor(
            tables=[
                TabularTableSpec(
                    name="departments",
                    primary_key="dept_id",
                    dataframe=departments_dataframe,
                ),
                TabularTableSpec(
                    name="employees",
                    primary_key="emp_id",
                    dataframe=employees_dataframe,
                    foreign_keys=[
                        ForeignKeySpec(
                            source_columns="dept_id",
                            target_table="departments",
                            target_columns="dept_id",
                            relation_label="BELONGS_TO_DEPARTMENT",
                        )
                    ],
                ),
            ]
        )
        documents = extractor.load_documents()

        # when / then
        with pytest.raises(ValueError, match="references missing row"):
            extractor(documents)
