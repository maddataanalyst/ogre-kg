"""Tabular extractors for row-to-graph conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph.transformations.dynamic_llm import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.schema import BaseNode, Document, TransformComponent

ROW_TABLE_NAME_KEY = "ogre_kg_table_name"
ROW_DATA_KEY = "ogre_kg_row_data"
ROW_SOURCE_PATH_KEY = "ogre_kg_source_path"


@dataclass(frozen=True)
class ForeignKeySpec:
    """Configuration describing a foreign-key relationship.

    Parameters
    ----------
    source_columns
        Source column or columns on the current table.
    target_table
        Referenced table name.
    target_columns
        Target column or columns on the referenced table. In v1 these must
        match the target table primary key.
    relation_label
        Graph edge label to create for the relationship.
    """

    source_columns: str | list[str]
    target_table: str
    target_columns: str | list[str]
    relation_label: str


@dataclass(frozen=True)
class RelationColumnSpec:
    """Configuration for a non-foreign-key relation column.

    Parameters
    ----------
    column
        Column name whose values should become related entities.
    relation_label
        Optional edge label override. Defaults to ``HAS_<COLUMN>``.
    target_label
        Optional target entity label override. Defaults to normalized column name.
    split_separator
        Optional separator indicating that one cell should be split into
        multiple related values.
    strip_values
        Whether split values should be whitespace-trimmed before entity creation.
    """

    column: str
    relation_label: str | None = None
    target_label: str | None = None
    split_separator: str | None = None
    strip_values: bool = True


@dataclass(frozen=True)
class TabularTableSpec:
    """Configuration describing how one tabular dataset should be converted.

    Parameters
    ----------
    name
        Logical table name used for row-entity identifiers.
    primary_key
        Primary-key column or columns for the table.
    dataframe
        Optional in-memory pandas DataFrame source.
    path
        Optional path to a CSV file.
    stream
        Optional in-memory text stream containing CSV data.
    entity_label
        Optional main entity label. Defaults to normalized table name.
    property_columns
        Columns that should always be stored as properties on the main entity.
    foreign_keys
        Explicit foreign-key relationships for the table.
    relation_columns
        Explicit non-foreign-key relation column overrides.
    ignored_columns
        Columns to skip entirely.
    """

    name: str
    primary_key: str | list[str]
    dataframe: pd.DataFrame | None = None
    path: str | Path | None = None
    stream: TextIO | None = None
    entity_label: str | None = None
    property_columns: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeySpec] = field(default_factory=list)
    relation_columns: list[RelationColumnSpec] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)


class TabularPathExtractor(TransformComponent):
    """Convert tabular rows into graph entities and relations.

    Each input row becomes one main entity identified by ``table + primary key``.
    Primary-key columns are stored on the main entity. Explicit foreign keys
    link to row entities in other configured tables. Scalar columns become
    properties by default, while remaining text columns become linked entities.

    Parameters
    ----------
    tables
        Table specifications defining tabular inputs and extraction behavior.
    """

    tables: list[TabularTableSpec]

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "TabularPathExtractor"

    def load_documents(self) -> list[Document]:
        """Load configured tabular rows into ``Document`` objects.

        Returns
        -------
        list[Document]
            One document per tabular row with row metadata attached.
        """
        documents: list[Document] = []

        for table in self.tables:
            rows = self._read_table_rows(table)
            source_descriptor = _source_descriptor(table)
            for row in rows:
                row_text = self._build_row_text(table.name, row)
                documents.append(
                    Document(
                        text=row_text,
                        metadata={
                            ROW_TABLE_NAME_KEY: table.name,
                            ROW_DATA_KEY: row,
                            ROW_SOURCE_PATH_KEY: source_descriptor,
                        },
                    )
                )

        return documents

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        """Attach graph metadata for tabular rows to input nodes.

        Parameters
        ----------
        nodes
            Documents or nodes containing row metadata produced by ``load_documents``.

        Returns
        -------
        list[BaseNode]
            Input nodes annotated with graph entities and relations.
        """
        del kwargs

        table_specs = {table.name: table for table in self.tables}
        table_rows = self._collect_rows_by_table(nodes, table_specs)
        table_indexes = self._build_table_indexes(table_specs, table_rows)

        transformed_nodes: list[BaseNode] = []
        for node in nodes:
            table_name, row = self._extract_row_metadata(node)
            table = table_specs[table_name]
            entity_nodes, relations = self._build_row_graph(table, row, table_specs, table_indexes)

            node.metadata[KG_NODES_KEY] = entity_nodes
            node.metadata[KG_RELATIONS_KEY] = relations
            transformed_nodes.append(node)

        return transformed_nodes

    def _read_table_rows(self, table: TabularTableSpec) -> list[dict[str, Any]]:
        """Read rows from one tabular source."""
        dataframe = self._load_table_dataframe(table)
        records = dataframe.to_dict(orient="records")
        return [dict(record) for record in records]

    def _load_table_dataframe(self, table: TabularTableSpec) -> pd.DataFrame:
        """Load a table source into a pandas DataFrame."""
        _validate_table_source(table)

        if table.dataframe is not None:
            dataframe = table.dataframe.copy(deep=True)
        elif table.path is not None:
            dataframe = pd.read_csv(Path(table.path))
        else:
            assert table.stream is not None
            if hasattr(table.stream, "seek"):
                table.stream.seek(0)
            dataframe = pd.read_csv(table.stream)

        return self._prepare_table_dataframe(dataframe, table)

    def _prepare_table_dataframe(
        self,
        dataframe: pd.DataFrame,
        table: TabularTableSpec,
    ) -> pd.DataFrame:
        """Normalize a DataFrame before row-to-graph conversion."""
        prepared_dataframe = dataframe.copy(deep=True)

        for relation_spec in table.relation_columns:
            if relation_spec.column not in prepared_dataframe.columns:
                continue
            prepared_dataframe[relation_spec.column] = prepared_dataframe[
                relation_spec.column
            ].apply(
                lambda value, current_spec=relation_spec: _normalize_relation_cell(
                    value, current_spec
                )
            )

        return prepared_dataframe

    def _collect_rows_by_table(
        self,
        nodes: list[BaseNode],
        table_specs: dict[str, TabularTableSpec],
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect row dicts from node metadata grouped by table."""
        rows_by_table = {table_name: [] for table_name in table_specs}

        for node in nodes:
            table_name, row = self._extract_row_metadata(node)
            rows_by_table[table_name].append(row)

        return rows_by_table

    def _build_table_indexes(
        self,
        table_specs: dict[str, TabularTableSpec],
        table_rows: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[tuple[str, ...], dict[str, Any]]]:
        """Build per-table indexes for row lookups by primary key."""
        table_indexes: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}

        for table_name, table in table_specs.items():
            rows = table_rows.get(table_name, [])
            self._validate_table_configuration(table, table_specs, rows)
            primary_key_columns = _ensure_list(table.primary_key)

            table_index: dict[tuple[str, ...], dict[str, Any]] = {}
            for row in rows:
                primary_key = self._make_key(row, primary_key_columns, context=table_name)
                if primary_key in table_index:
                    raise ValueError(
                        f"Duplicate primary key {primary_key!r} found in table '{table_name}'."
                    )
                table_index[primary_key] = row

            table_indexes[table_name] = table_index

        return table_indexes

    def _validate_table_configuration(
        self,
        table: TabularTableSpec,
        table_specs: dict[str, TabularTableSpec],
        rows: list[dict[str, Any]],
    ) -> None:
        """Validate one table specification against loaded rows."""
        _validate_table_source(table)

        relation_columns = {spec.column for spec in table.relation_columns}
        property_columns = set(table.property_columns)
        ignored_columns = set(table.ignored_columns)
        foreign_key_columns = {
            column
            for foreign_key in table.foreign_keys
            for column in _ensure_list(foreign_key.source_columns)
        }

        conflicts = (
            (property_columns & ignored_columns)
            | (property_columns & relation_columns)
            | (ignored_columns & relation_columns)
            | (foreign_key_columns & ignored_columns)
            | (foreign_key_columns & property_columns)
            | (foreign_key_columns & relation_columns)
        )
        if conflicts:
            raise ValueError(
                f"Columns {sorted(conflicts)!r} in table '{table.name}' are configured "
                "with conflicting behaviors."
            )

        available_columns = set(rows[0]) if rows else set()
        required_columns = (
            set(_ensure_list(table.primary_key))
            | property_columns
            | ignored_columns
            | relation_columns
            | foreign_key_columns
        )

        missing_columns = required_columns - available_columns
        if rows and missing_columns:
            raise ValueError(
                f"Columns {sorted(missing_columns)!r} are not present in table '{table.name}'."
            )

        for foreign_key in table.foreign_keys:
            target_table = table_specs.get(foreign_key.target_table)
            if target_table is None:
                raise ValueError(
                    f"Foreign key on table '{table.name}' references unknown table "
                    f"'{foreign_key.target_table}'."
                )

            source_columns = _ensure_list(foreign_key.source_columns)
            target_columns = _ensure_list(foreign_key.target_columns)
            target_primary_key = _ensure_list(target_table.primary_key)

            if len(source_columns) != len(target_columns):
                raise ValueError(
                    f"Foreign key '{foreign_key.relation_label}' on table '{table.name}' "
                    "must use the same number of source and target columns."
                )

            if target_columns != target_primary_key:
                raise ValueError(
                    f"Foreign key '{foreign_key.relation_label}' on table '{table.name}' must "
                    f"target the primary key of table '{target_table.name}'."
                )

    def _extract_row_metadata(self, node: BaseNode) -> tuple[str, dict[str, Any]]:
        """Extract table row metadata from an input node."""
        table_name = node.metadata.get(ROW_TABLE_NAME_KEY)
        row = node.metadata.get(ROW_DATA_KEY)

        if not isinstance(table_name, str):
            raise ValueError(
                f"Node metadata must include string key '{ROW_TABLE_NAME_KEY}' for tabular "
                "extraction."
            )
        if not isinstance(row, dict):
            raise ValueError(
                f"Node metadata must include dict key '{ROW_DATA_KEY}' for tabular extraction."
            )

        return table_name, row

    def _build_row_graph(
        self,
        table: TabularTableSpec,
        row: dict[str, Any],
        table_specs: dict[str, TabularTableSpec],
        table_indexes: dict[str, dict[tuple[str, ...], dict[str, Any]]],
    ) -> tuple[list[EntityNode], list[Relation]]:
        """Build graph entities and relations for one row."""
        main_entity = self._build_main_entity(table, row)
        entity_nodes = [main_entity]
        relations: list[Relation] = []

        property_columns = set(table.property_columns)
        ignored_columns = set(table.ignored_columns)
        relation_specs = {spec.column: spec for spec in table.relation_columns}
        foreign_key_by_column = {
            column: foreign_key
            for foreign_key in table.foreign_keys
            for column in _ensure_list(foreign_key.source_columns)
        }
        primary_key_columns = set(_ensure_list(table.primary_key))

        for column_name, raw_value in row.items():
            if column_name in primary_key_columns or column_name in ignored_columns:
                continue
            if _is_empty(raw_value):
                continue

            foreign_key = foreign_key_by_column.get(column_name)
            if foreign_key is not None:
                if column_name != _ensure_list(foreign_key.source_columns)[0]:
                    continue
                target_entity, relation = self._build_foreign_key_relation(
                    table=table,
                    row=row,
                    main_entity=main_entity,
                    foreign_key=foreign_key,
                    table_specs=table_specs,
                    table_indexes=table_indexes,
                )
                entity_nodes.append(target_entity)
                relations.append(relation)
                continue

            relation_spec = relation_specs.get(column_name)
            if relation_spec is not None:
                relation_values = _split_relation_values(raw_value, relation_spec)
                relation_label = (
                    relation_spec.relation_label
                    if relation_spec.relation_label is not None
                    else _default_relation_label(column_name)
                )

                for relation_value in relation_values:
                    target_entity = self._build_value_entity(
                        column_name,
                        relation_value,
                        relation_spec,
                    )
                    entity_nodes.append(target_entity)
                    relations.append(
                        Relation(
                            label=relation_label,
                            source_id=main_entity.name,
                            target_id=target_entity.name,
                            properties={"source_column": column_name},
                        )
                    )
                continue

            if column_name in property_columns or self._should_treat_as_property(raw_value):
                main_entity.properties[column_name] = _coerce_property_value(raw_value)
                continue

            target_entity = self._build_value_entity(column_name, raw_value, relation_spec)
            relation_label = (
                relation_spec.relation_label
                if relation_spec is not None and relation_spec.relation_label is not None
                else _default_relation_label(column_name)
            )

            entity_nodes.append(target_entity)
            relations.append(
                Relation(
                    label=relation_label,
                    source_id=main_entity.name,
                    target_id=target_entity.name,
                    properties={"source_column": column_name},
                )
            )

        return _deduplicate_entity_nodes(entity_nodes), relations

    def _build_main_entity(self, table: TabularTableSpec, row: dict[str, Any]) -> EntityNode:
        """Build the main row entity for a table row."""
        primary_key_columns = _ensure_list(table.primary_key)
        entity_name = _row_entity_name(table.name, primary_key_columns, row)
        entity_label = table.entity_label or _normalize_label(table.name)

        properties = {
            column: _coerce_property_value(row[column])
            for column in primary_key_columns
            if not _is_empty(row.get(column))
        }

        return EntityNode(name=entity_name, label=entity_label, properties=properties)

    def _build_foreign_key_relation(
        self,
        table: TabularTableSpec,
        row: dict[str, Any],
        main_entity: EntityNode,
        foreign_key: ForeignKeySpec,
        table_specs: dict[str, TabularTableSpec],
        table_indexes: dict[str, dict[tuple[str, ...], dict[str, Any]]],
    ) -> tuple[EntityNode, Relation]:
        """Build one foreign-key relation from the current row."""
        source_columns = _ensure_list(foreign_key.source_columns)
        target_table = table_specs[foreign_key.target_table]
        target_primary_key = _ensure_list(target_table.primary_key)
        target_key = self._make_key(row, source_columns, context=table.name)
        target_row = table_indexes[foreign_key.target_table].get(target_key)

        if target_row is None:
            raise ValueError(
                f"Foreign key '{foreign_key.relation_label}' on table '{table.name}' references "
                f"missing row in table '{foreign_key.target_table}' for key {target_key!r}."
            )

        target_entity = self._build_main_entity(target_table, target_row)
        relation = Relation(
            label=foreign_key.relation_label,
            source_id=main_entity.name,
            target_id=target_entity.name,
            properties={
                "source_columns": source_columns,
                "target_table": foreign_key.target_table,
                "target_columns": target_primary_key,
            },
        )

        return target_entity, relation

    def _build_value_entity(
        self,
        column_name: str,
        raw_value: Any,
        relation_spec: RelationColumnSpec | None,
    ) -> EntityNode:
        """Build a value entity for a non-FK relational column."""
        target_label = (
            relation_spec.target_label
            if relation_spec is not None and relation_spec.target_label is not None
            else _normalize_label(column_name)
        )
        value_name = _stringify_relation_value(raw_value)
        target_name = f"{target_label}:{value_name}"
        return EntityNode(
            name=target_name,
            label=target_label,
            properties={"value": _coerce_property_value(raw_value), "source_column": column_name},
        )

    def _make_key(
        self,
        row: dict[str, Any],
        columns: list[str],
        context: str,
    ) -> tuple[str, ...]:
        """Build a composite key from row columns."""
        key_values: list[str] = []
        for column in columns:
            if column not in row:
                raise ValueError(f"Column '{column}' is missing in table '{context}'.")

            value = row[column]
            if _is_empty(value):
                raise ValueError(
                    f"Column '{column}' in table '{context}' cannot be empty when used in a key."
                )
            key_values.append(_stringify_identifier_value(value))

        return tuple(key_values)

    def _should_treat_as_property(self, raw_value: Any) -> bool:
        """Return whether a value should be treated as a scalar property."""
        return _is_scalar_value(raw_value)

    @staticmethod
    def _build_row_text(table_name: str, row: dict[str, Any]) -> str:
        """Build a readable text representation for a row document."""
        row_lines = [f"Table: {table_name}"]
        row_lines.extend(f"{column}: {value}" for column, value in row.items())
        return "\n".join(row_lines)


def _validate_table_source(table: TabularTableSpec) -> None:
    """Validate that a table spec has exactly one defined source."""
    source_count = sum(source is not None for source in (table.dataframe, table.path, table.stream))
    if source_count != 1:
        raise ValueError(
            f"Table '{table.name}' must define exactly one source among "
            "'dataframe', 'path', or 'stream'."
        )


def _source_descriptor(table: TabularTableSpec) -> str:
    """Return a human-readable source description for metadata."""
    if table.path is not None:
        return str(table.path)
    if table.stream is not None:
        return "<stream>"
    return "<dataframe>"


def _ensure_list(value: str | list[str]) -> list[str]:
    """Normalize a scalar-or-list config field into a list."""
    if isinstance(value, list):
        return value
    return [value]


def _normalize_label(value: str) -> str:
    """Normalize a table or column name into a graph label."""
    tokens = value.replace("-", "_").split("_")
    return "".join(token.capitalize() for token in tokens if token)


def _default_relation_label(column_name: str) -> str:
    """Generate a default relation label for a column."""
    return f"HAS_{column_name.replace('-', '_').upper()}"


def _normalize_relation_cell(value: Any, relation_spec: RelationColumnSpec) -> Any:
    """Normalize relation-cell values before row-to-graph conversion."""
    if _is_empty(value):
        return None
    return _split_relation_values(value, relation_spec)


def _split_relation_values(raw_value: Any, relation_spec: RelationColumnSpec) -> list[Any]:
    """Split and normalize configured relation values from one cell."""
    if isinstance(raw_value, list):
        values = list(raw_value)
    elif relation_spec.split_separator is None:
        values = [raw_value]
    else:
        values = str(raw_value).split(relation_spec.split_separator)

    if relation_spec.strip_values:
        values = [value.strip() if isinstance(value, str) else value for value in values]

    normalized_values = [value for value in values if not _is_empty(value)]

    deduplicated_values: list[Any] = []
    seen_values: set[str] = set()
    for value in normalized_values:
        value_key = _stringify_identifier_value(value)
        if value_key in seen_values:
            continue
        seen_values.add(value_key)
        deduplicated_values.append(value)

    return deduplicated_values


def _row_entity_name(
    table_name: str,
    primary_key_columns: list[str],
    row: dict[str, Any],
) -> str:
    """Build a stable row-entity identifier from table name and key values."""
    key_parts = [
        f"{column}={_stringify_identifier_value(row[column])}" for column in primary_key_columns
    ]
    return f"{table_name}:{'|'.join(key_parts)}"


def _is_empty(value: Any) -> bool:
    """Return whether a raw tabular value should be considered empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(result, bool):
        return result

    return False


def _is_scalar_value(raw_value: Any) -> bool:
    """Return whether a value should be treated as a scalar property."""
    if _is_empty(raw_value):
        return False
    if isinstance(raw_value, bool | int | float | date | datetime | pd.Timestamp):
        return True
    if not isinstance(raw_value, str):
        return False

    value = raw_value.strip()
    lowered = value.lower()
    if lowered in {"true", "false", "yes", "no"}:
        return True

    try:
        int(value)
        return True
    except ValueError:
        pass

    try:
        float(value)
        return True
    except ValueError:
        pass

    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        pass

    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _coerce_property_value(raw_value: Any) -> Any:
    """Coerce a tabular value into a graph property value."""
    if isinstance(raw_value, pd.Timestamp):
        return raw_value.isoformat()
    if isinstance(raw_value, datetime):
        return raw_value.isoformat()
    if isinstance(raw_value, date):
        return raw_value.isoformat()
    if isinstance(raw_value, bool | int | float):
        return raw_value
    if not isinstance(raw_value, str):
        return raw_value

    value = raw_value.strip()
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _stringify_identifier_value(value: Any) -> str:
    """Convert a value into a stable string form for entity identifiers."""
    coerced_value = _coerce_property_value(value)
    if isinstance(coerced_value, float) and coerced_value.is_integer():
        return str(int(coerced_value))
    return str(coerced_value).strip()


def _stringify_relation_value(value: Any) -> str:
    """Convert a relation value into a stable string form for entity names."""
    coerced_value = _coerce_property_value(value)
    if isinstance(coerced_value, float) and coerced_value.is_integer():
        return str(int(coerced_value))
    if isinstance(value, str):
        return value
    return str(coerced_value)


def _deduplicate_entity_nodes(entity_nodes: list[EntityNode]) -> list[EntityNode]:
    """Deduplicate entity nodes by ``(name, label)`` while preserving order."""
    deduplicated_nodes: list[EntityNode] = []
    seen_nodes: set[tuple[str, str]] = set()

    for entity_node in entity_nodes:
        node_key = (entity_node.name, entity_node.label)
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key)
        deduplicated_nodes.append(entity_node)

    return deduplicated_nodes
