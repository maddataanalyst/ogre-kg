"""Tests for ogre_kg.utils module."""

from __future__ import annotations

import pytest
from conftest import FakeMemgraphStore, FakeNeo4jStore, FakeStructuredStore

from ogre_kg.utils import (
    GraphStoreBackend,
    UnionFind,
    build_entity_groups,
    detect_graph_store_backend,
    quote_cypher,
    resolve_graph_store,
)


class TestUnionFind:
    def test_find_creates_element_on_access(self):
        # given
        uf = UnionFind()

        # when
        root = uf.find("A")

        # then
        assert root == "A"

    def test_union_merges_two_elements(self):
        # given
        uf = UnionFind()
        uf.union("A", "B")

        # when
        groups = uf.groups(min_size=1)

        # then
        assert len(groups) == 1
        assert {"A", "B"} in groups

    def test_transitive_union(self):
        # given
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")

        # when
        groups = uf.groups()

        # then
        assert len(groups) == 1
        assert {"A", "B", "C"} in groups

    def test_groups_skips_singletons_by_default(self):
        # given
        uf = UnionFind(["A", "B", "C"])
        uf.union("A", "B")

        # when
        groups = uf.groups()

        # then
        assert len(groups) == 1
        assert {"A", "B"} in groups

    def test_groups_min_size_1_includes_singletons(self):
        # given
        uf = UnionFind(["X"])

        # when
        groups = uf.groups(min_size=1)

        # then
        assert len(groups) == 1
        assert {"X"} in groups

    def test_init_with_elements(self):
        # given / when
        uf = UnionFind(["A", "B", "C"])

        # then
        assert uf.find("A") == "A"
        assert uf.find("B") == "B"
        assert uf.find("C") == "C"

    def test_path_compression(self):
        # given - create a chain A -> B -> C -> D
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        uf.union("C", "D")

        # when - finding A should compress the path
        root = uf.find("A")

        # then - all elements share the same root
        assert uf.find("B") == root
        assert uf.find("C") == root
        assert uf.find("D") == root


class TestBuildEntityGroups:
    def test_simple_pairs(self):
        # given
        pairs = [
            {"name1": "A", "name2": "B"},
            {"name1": "C", "name2": "D"},
        ]

        # when
        groups = build_entity_groups(pairs)

        # then
        assert len(groups) == 2
        assert {"A", "B"} in groups
        assert {"C", "D"} in groups

    def test_transitive_closure(self):
        # given - A-B and B-C should form one group {A, B, C}
        pairs = [
            {"name1": "A", "name2": "B"},
            {"name1": "B", "name2": "C"},
        ]

        # when
        groups = build_entity_groups(pairs)

        # then
        assert len(groups) == 1
        assert {"A", "B", "C"} in groups

    def test_longer_chain(self):
        # given
        pairs = [
            {"name1": "A", "name2": "B"},
            {"name1": "C", "name2": "D"},
            {"name1": "B", "name2": "C"},
        ]

        # when
        groups = build_entity_groups(pairs)

        # then
        assert len(groups) == 1
        assert {"A", "B", "C", "D"} in groups

    def test_empty_pairs(self):
        # given / when
        groups = build_entity_groups([])

        # then
        assert groups == []

    def test_skips_invalid_pairs(self):
        # given
        pairs = [
            {"name1": "", "name2": "B"},
            {"name1": "C", "name2": None},
            {"name1": 42, "name2": "D"},
            {"name1": "E", "name2": "F"},
        ]

        # when
        groups = build_entity_groups(pairs)

        # then
        assert len(groups) == 1
        assert {"E", "F"} in groups

    def test_duplicate_pairs_handled(self):
        # given
        pairs = [
            {"name1": "A", "name2": "B"},
            {"name1": "A", "name2": "B"},
            {"name1": "B", "name2": "A"},
        ]

        # when
        groups = build_entity_groups(pairs)

        # then
        assert len(groups) == 1
        assert {"A", "B"} in groups


class TestQuoteCypher:
    def test_plain_string(self):
        assert quote_cypher("hello") == "hello"

    def test_escapes_single_quotes(self):
        assert quote_cypher("O'Brien") == "O\\'Brien"

    def test_escapes_backslashes(self):
        assert quote_cypher("path\\to") == "path\\\\to"

    def test_escapes_both(self):
        assert quote_cypher("it's a\\path") == "it\\'s a\\\\path"


class TestDetectGraphStoreBackend:
    def test_detects_memgraph(self):
        # given
        store = FakeMemgraphStore()

        # when
        backend = detect_graph_store_backend(store)

        # then
        assert backend == GraphStoreBackend.MEMGRAPH

    def test_detects_neo4j(self):
        # given
        store = FakeNeo4jStore()

        # when
        backend = detect_graph_store_backend(store)

        # then
        assert backend == GraphStoreBackend.NEO4J

    def test_raises_for_unknown_backend(self):
        # given
        store = FakeStructuredStore()

        # when / then
        with pytest.raises(ValueError, match="Cannot detect backend"):
            detect_graph_store_backend(store)


class TestResolveGraphStore:
    def test_resolves_direct_store(self, fake_store):
        # when
        result = resolve_graph_store(fake_store)

        # then
        assert result is fake_store

    def test_resolves_index_like_object(self, fake_store, fake_index):
        # when
        result = resolve_graph_store(fake_index)

        # then
        assert result is fake_store

    def test_raises_for_invalid_source(self):
        # given / when / then
        with pytest.raises(ValueError, match="Expected a structured-query"):
            resolve_graph_store("not a store")
