"""
Tests for Evidence Intelligence 2.0 (Phase 11).

Verifies:
1. Ten copied sources != ten independent confirmations (lineage collapse)
2. Declared and observed dependence both merge lineages
3. Independence-weighted consensus math
4. Verdict mapping (insufficient / single-lineage / verified / contested)
5. Quality decomposition (reliability, independence, corroboration, recency)
6. Scaffolding honesty markers
"""

import pytest
import time

from sweep_cognitive.evidence2 import (
    SourceRegistry,
    EvidenceEngine2,
    Source,
    SourceItem,
    shingle_text,
    shingle_jaccard,
    shingle_containment,
)


WIRE_STORY = (
    "Researchers at a major university announced a breakthrough in "
    "solid-state battery technology on Monday, claiming doubled energy "
    "density and tripled charge cycles in independent lab tests."
)


class TestShingling:
    def test_identical_text_full_overlap(self):
        a = shingle_text("the quick brown fox jumps over the lazy dog")
        b = shingle_text("the quick brown fox jumps over the lazy dog")
        assert shingle_jaccard(a, b) == 1.0

    def test_disjoint_text_no_overlap(self):
        a = shingle_text("alpha beta gamma delta epsilon")
        b = shingle_text("one two three four five")
        assert shingle_jaccard(a, b) == 0.0

    def test_copied_text_high_containment(self):
        original = shingle_text(WIRE_STORY)
        # Rewritten slightly (one word changed) — copy detection uses
        # containment, which is robust to single-word edits.
        rewritten = shingle_text(WIRE_STORY.replace("Monday", "Tuesday"))
        assert shingle_containment(original, rewritten) > 0.8

    def test_empty_safe(self):
        assert shingle_jaccard(set(), shingle_text("x")) == 0.0


class TestSourceRegistry:
    def test_register_and_get(self):
        reg = SourceRegistry()
        reg.register_source("wire", "Wire Service", reliability=0.9)
        src = reg.get_source("wire")
        assert src is not None
        assert src.reliability == 0.9

    def test_unknown_source_item_rejected(self):
        reg = SourceRegistry()
        with pytest.raises(KeyError):
            reg.add_item("ghost", "content")

    def test_declared_dependence_requires_both(self):
        reg = SourceRegistry()
        reg.register_source("a")
        with pytest.raises(KeyError):
            reg.add_declared_dependence("a", "missing")

    def test_declared_dependence_creates_edge(self):
        reg = SourceRegistry()
        reg.register_source("upstream")
        reg.register_source("downstream")
        reg.add_declared_dependence("upstream", "downstream")
        assert reg.stats()["declared_edges"] == 1
        assert reg.effective_independent_sources(["upstream", "downstream"]) == 1

    def test_stance_clamped(self):
        reg = SourceRegistry()
        reg.register_source("s")
        item = reg.add_item("s", "text", stance=5.0)
        assert item.stance == 1.0
        item2 = reg.add_item("s", "text2", stance=-5.0)
        assert item2.stance == -1.0


class TestLineageCollapse:
    """The core spec requirement: copied sources don't stack."""

    def _registry_with_syndication(self, n_outlets: int = 5) -> SourceRegistry:
        reg = SourceRegistry()
        reg.register_source("wire", "Wire", reliability=0.9)
        for i in range(n_outlets):
            outlet = f"outlet_{i}"
            reg.register_source(outlet, outlet, reliability=0.6)
            reg.add_declared_dependence("wire", outlet)
        return reg

    def test_ten_copies_are_one_lineage(self):
        reg = self._registry_with_syndication(9)
        sources = ["wire"] + [f"outlet_{i}" for i in range(9)]
        assert reg.effective_independent_sources(sources) == 1

    def test_transitive_dependence(self):
        """A copies B, B copies C → A, B, C all one lineage."""
        reg = SourceRegistry()
        for sid in ("a", "b", "c"):
            reg.register_source(sid)
        reg.add_declared_dependence("c", "b")
        reg.add_declared_dependence("b", "a")
        assert reg.effective_independent_sources(["a", "b", "c"]) == 1

    def test_independent_sources_stay_separate(self):
        reg = SourceRegistry()
        for sid in ("x", "y", "z"):
            reg.register_source(sid)
        assert reg.effective_independent_sources(["x", "y", "z"]) == 3

    def test_lineage_groups_expose_partitions(self):
        reg = self._registry_with_syndication(2)
        groups = reg.lineage_groups(["wire", "outlet_0", "outlet_1", "independent"])
        reg.register_source("independent")
        groups = reg.lineage_groups(["wire", "outlet_0", "outlet_1", "independent"])
        assert len(groups) == 2
        group_sizes = sorted(len(g) for g in groups)
        assert group_sizes == [1, 3]

    def test_observed_dependence_via_content_hash(self):
        """Same content hash + time gap = copied wire story."""
        reg = SourceRegistry()
        reg.register_source("news1")
        reg.register_source("news2")
        t0 = time.time()
        i1 = reg.add_item("news1", WIRE_STORY, content_hash="abc123", timestamp=t0)
        i2 = reg.add_item("news2", WIRE_STORY, content_hash="abc123", timestamp=t0 + 3600)
        found = reg.observe_dependence_among([i1.item_id, i2.item_id])
        assert len(found) == 1
        assert found[0]["kind"] == "same_content_hash"
        assert reg.effective_independent_sources(["news1", "news2"]) == 1

    def test_same_content_same_time_not_dependence(self):
        """Identical content published simultaneously is ambiguous —
        could be a shared press release quoted independently. No edge."""
        reg = SourceRegistry()
        reg.register_source("news1")
        reg.register_source("news2")
        t0 = time.time()
        i1 = reg.add_item("news1", WIRE_STORY, content_hash="abc123", timestamp=t0)
        i2 = reg.add_item("news2", WIRE_STORY, content_hash="abc123", timestamp=t0)
        found = reg.observe_dependence_among([i1.item_id, i2.item_id])
        assert found == []
        assert reg.effective_independent_sources(["news1", "news2"]) == 2

    def test_observed_dependence_via_near_duplicate(self):
        reg = SourceRegistry()
        reg.register_source("blog1")
        reg.register_source("blog2")
        t0 = time.time()
        i1 = reg.add_item("blog1", WIRE_STORY, timestamp=t0)
        i2 = reg.add_item("blog2", WIRE_STORY.replace("Monday", "Tuesday"), timestamp=t0 + 7200)
        found = reg.observe_dependence_among([i1.item_id, i2.item_id])
        assert len(found) == 1
        assert found[0]["kind"] == "near_duplicate_text"
        assert reg.effective_independent_sources(["blog1", "blog2"]) == 1

    def test_distinct_content_no_dependence(self):
        reg = SourceRegistry()
        reg.register_source("one")
        reg.register_source("two")
        t0 = time.time()
        i1 = reg.add_item("one", "Cats are mammals that purr loudly at night", timestamp=t0)
        i2 = reg.add_item("two", "Stocks fell sharply amid rate concerns today", timestamp=t0 + 60)
        found = reg.observe_dependence_among([i1.item_id, i2.item_id])
        assert found == []
        assert reg.effective_independent_sources(["one", "two"]) == 2

    def test_same_source_never_dependence(self):
        reg = SourceRegistry()
        reg.register_source("solo")
        i1 = reg.add_item("solo", WIRE_STORY, content_hash="same")
        i2 = reg.add_item("solo", WIRE_STORY, content_hash="same")
        assert reg.detect_shared_origin(i1, i2) is None


class TestConsensus:
    def _setup_verified_case(self):
        """Two genuinely independent sources supporting the same claim."""
        reg = SourceRegistry()
        reg.register_source("journal", "Journal", reliability=0.95)
        reg.register_source("govdb", "Gov database", reliability=0.9)
        i1 = reg.add_item("journal", "Study confirms claim X with data", stance=0.9)
        i2 = reg.add_item("govdb", "Official records confirm claim X", stance=0.8)
        return reg, [i1.item_id, i2.item_id]

    def _setup_copied_case(self):
        """Five syndicated copies of one wire story."""
        reg = SourceRegistry()
        reg.register_source("wire", reliability=0.9)
        for i in range(4):
            reg.register_source(f"outlet_{i}", reliability=0.6)
            reg.add_declared_dependence("wire", f"outlet_{i}")
        ids = [reg.add_item("wire", WIRE_STORY, stance=0.9).item_id]
        for i in range(4):
            ids.append(reg.add_item(f"outlet_{i}", WIRE_STORY, stance=0.9).item_id)
        return reg, ids

    def test_verified_tendency_with_independence(self):
        reg, ids = self._setup_verified_case()
        engine = EvidenceEngine2(reg)
        result = engine.consensus(ids, claim="X is true")
        assert result["verdict"] == "VERIFIED_TENDENCY"
        assert result["effective_sources"] == 2
        assert result["support_weight"] > 0
        assert result["oppose_weight"] == 0

    def test_copied_sources_yield_single_lineage(self):
        reg, ids = self._setup_copied_case()
        engine = EvidenceEngine2(reg)
        result = engine.consensus(ids, claim="battery breakthrough")
        assert result["effective_sources"] == 1
        assert result["verdict"] == "SINGLE_LINEAGE_ONLY"
        # The five copies must weigh far less than five independent items
        # would. The lineage factor caps the copied contribution.
        copied_total = result["support_weight"]
        # A single strong item with quality ~0.85 * 1.0 lineage factor
        # should be roughly the ceiling for one lineage.
        assert copied_total <= 1.0

    def test_copies_weigh_less_than_independents(self):
        """Same stance count, but copies contribute less than independent
        sources would (lineage division)."""
        reg_c, ids_c = self._setup_copied_case()
        reg_i = SourceRegistry()
        ids_i = []
        t0 = time.time()
        for i in range(5):
            reg_i.register_source(f"src_{i}", reliability=0.6)
            # Independent publications: identical press-release text but
            # distinct publish times, gaps > 60s would wrongly imply copying,
            # so we use same-timestamp ingestion (shared quotation case) —
            # the registry must NOT merge these.
            ids_i.append(reg_i.add_item(
                f"src_{i}", WIRE_STORY, stance=0.9, timestamp=t0
            ).item_id)
        engine_c = EvidenceEngine2(reg_c)
        engine_i = EvidenceEngine2(reg_i)
        r_c = engine_c.consensus(ids_c, claim="c")
        r_i = engine_i.consensus(ids_i, claim="i")
        # Both have 5 items; only the copied case should be lineage-collapsed
        assert r_c["effective_sources"] == 1
        assert r_i["effective_sources"] == 5
        assert r_c["support_weight"] < r_i["support_weight"]

    def test_contested_verdict(self):
        reg = SourceRegistry()
        reg.register_source("pro", reliability=0.8)
        reg.register_source("con", reliability=0.8)
        i1 = reg.add_item("pro", "Evidence supports the claim", stance=0.8)
        i2 = reg.add_item("con", "Evidence contradicts the claim", stance=-0.8)
        engine = EvidenceEngine2(reg)
        result = engine.consensus([i1.item_id, i2.item_id], claim="X")
        assert result["verdict"] == "CONTESTED"
        assert result["contradiction_severity"] > 0.4

    def test_refuted_tendency(self):
        reg = SourceRegistry()
        reg.register_source("a")
        reg.register_source("b")
        i1 = reg.add_item("a", "disproof one", stance=-0.9)
        i2 = reg.add_item("b", "disproof two", stance=-0.9)
        engine = EvidenceEngine2(reg)
        result = engine.consensus([i1.item_id, i2.item_id], claim="X")
        assert result["verdict"] == "REFUTED_TENDENCY"

    def test_insufficient_evidence(self):
        reg = SourceRegistry()
        reg.register_source("a")
        engine = EvidenceEngine2(reg)
        result = engine.consensus([], claim="X")
        assert result["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert result["effective_sources"] == 0

    def test_unknown_item_raises(self):
        reg = SourceRegistry()
        reg.register_source("a")
        engine = EvidenceEngine2(reg)
        with pytest.raises(KeyError):
            engine.consensus(["item_ghost"], claim="X")

    def test_neutral_items_no_support(self):
        reg = SourceRegistry()
        reg.register_source("a")
        reg.register_source("b")
        i1 = reg.add_item("a", "mentions topic", stance=0.0)
        i2 = reg.add_item("b", "mentions topic", stance=0.0)
        engine = EvidenceEngine2(reg)
        result = engine.consensus([i1.item_id, i2.item_id], claim="X")
        assert result["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert result["support_weight"] == 0.0


class TestQuality:
    def test_reliable_source_scores_higher(self):
        reg = SourceRegistry()
        reg.register_source("good", reliability=0.95)
        reg.register_source("bad", reliability=0.30)
        i_good = reg.add_item("good", "confirms X with data", stance=0.8)
        i_bad = reg.add_item("bad", "confirms X too", stance=0.8)
        engine = EvidenceEngine2(reg)
        q_good = engine.score_item(i_good.item_id)
        q_bad = engine.score_item(i_bad.item_id)
        assert q_good.overall > q_bad.overall
        assert q_good.source_reliability == 0.95

    def test_recency_decay(self):
        reg = SourceRegistry()
        reg.register_source("s")
        now = time.time()
        fresh = reg.add_item("s", "item one", stance=0.5, timestamp=now - 3600)
        old = reg.add_item("s", "item two", stance=0.5, timestamp=now - 60 * 86400)
        engine = EvidenceEngine2(reg)
        q_fresh = engine.score_item(fresh.item_id, now=now)
        q_old = engine.score_item(old.item_id, now=now)
        assert q_fresh.recency_factor > q_old.recency_factor
        # ~60 days ≈ 8.6 half-lives → recency < 0.01
        assert q_old.recency_factor < 0.01

    def test_corroboration_from_independent_sources(self):
        reg = SourceRegistry()
        reg.register_source("a")
        reg.register_source("b")
        i1 = reg.add_item("a", "supports claim", stance=0.8)
        i2 = reg.add_item("b", "also supports claim", stance=0.7)
        engine = EvidenceEngine2(reg)
        q1 = engine.score_item(i1.item_id)
        assert q1.corroboration_factor > 0  # b independently agrees

    def test_unknown_item_raises(self):
        reg = SourceRegistry()
        engine = EvidenceEngine2(reg)
        with pytest.raises(KeyError):
            engine.score_item("ghost")

    def test_quality_decomposition_sums_to_overall(self):
        reg = SourceRegistry()
        reg.register_source("s", reliability=0.8)
        i1 = reg.add_item("s", "some content", stance=0.5)
        engine = EvidenceEngine2(reg)
        q = engine.score_item(i1.item_id)
        expected = 0.35 * q.source_reliability + 0.25 * q.independence_factor \
            + 0.20 * q.corroboration_factor + 0.20 * q.recency_factor
        assert q.overall == pytest.approx(expected, abs=1e-9)


class TestScaffoldingHonesty:
    def test_duplicate_detection_marked_as_scaffold(self):
        import sweep_cognitive.evidence2 as mod
        doc = mod.__doc__ or ""
        assert "TEMPORARY SCAFFOLDING" in doc

    def test_shingle_docstring_declares_scope(self):
        assert "SCAFFOLDING" in shingle_text.__doc__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
