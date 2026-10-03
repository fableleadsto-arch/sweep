"""Unit tests for refactored cortex modules and web scraper.

Tests:
  - trace.py: ReasoningTrace, ReasoningResult
  - fast_path.py: try_fast_path, quick_direction
  - evidence_pipeline.py: cross_reference_evidence, apply_xref_adjustments
  - complexity.py: classify_query_complexity, select_reasoning_modules
  - human_reasoning.py: run_human_reasoning
  - web_scraper/: WebScraper, ContentExtractor, WebResearcher, PDFScraper, HeadlessBrowser
"""
import sys
import os
import time

# Ensure sweep_neural_mesh is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.signal import Signal, SignalType


# ══════════════════════════════════════════════════════════════
# trace.py
# ══════════════════════════════════════════════════════════════

def test_reasoning_trace():
    from neurons.trace import ReasoningTrace
    trace = ReasoningTrace(
        query="test", input_evidence_count=3,
        center_outputs={}, integration_confidence=0.8,
        decision="supported", decision_confidence=0.8,
        reasoning="because", total_latency_ms=1.0,
    )
    assert trace.query == "test"
    assert trace.input_evidence_count == 3
    assert trace.decision == "supported"
    print("  PASS: ReasoningTrace")


def test_reasoning_result():
    from neurons.trace import ReasoningResult, ReasoningTrace
    trace = ReasoningTrace(
        query="test", input_evidence_count=0,
        center_outputs={}, integration_confidence=0.8,
        decision="supported", decision_confidence=0.8,
        reasoning="because", total_latency_ms=1.0,
    )
    result = ReasoningResult(
        query="test", decision="supported", confidence=0.8,
        reasoning="because", explanation_data={}, trace=trace,
        factors=[{"name": "test", "score": 0.8}],
    )
    assert result.query == "test"
    assert result.decision == "supported"
    assert result.confidence == 0.8
    print("  PASS: ReasoningResult")


# ══════════════════════════════════════════════════════════════
# fast_path.py
# ══════════════════════════════════════════════════════════════

def test_fast_path():
    from neurons.fast_path import try_fast_path
    from neurons.world_knowledge import WorldKnowledge
    from neurons.trace import ReasoningTrace

    wk = WorldKnowledge()
    traces = []

    # Unanimous evidence
    evidence = [
        {"text": "Water makes things wet", "source": "test"},
        {"text": "Water is a liquid", "source": "test"},
    ]
    result = try_fast_path("Is water wet?", evidence, wk, time.time(), traces)
    assert result is not None
    assert result.decision in ("supported", "refuted", "insufficient")
    print("  PASS: try_fast_path")


def test_quick_direction():
    from neurons.fast_path import quick_direction
    # quick_direction takes a single text string
    d1 = quick_direction("Water makes things wet")
    d2 = quick_direction("The earth is flat and wrong")
    d3 = quick_direction("Some random text with no direction signals")
    assert d1 in ("supports", "refutes", "neutral")
    assert d2 in ("supports", "refutes", "neutral")
    assert d3 in ("supports", "refutes", "neutral")
    print(f"  PASS: quick_direction (wet='{d1}', flat='{d2}', neutral='{d3}')")


# ══════════════════════════════════════════════════════════════
# evidence_pipeline.py
# ══════════════════════════════════════════════════════════════

def test_cross_reference_evidence():
    from neurons.evidence_pipeline import cross_reference_evidence
    signals = [
        Signal(data={"evidence_text": "test"}, signal_type=SignalType.EVIDENCE, confidence=0.8, source_center="test"),
    ]
    credibility = [
        Signal(data={"credibility": 0.9}, signal_type=SignalType.CREDIBILITY, confidence=0.9, source_center="test"),
    ]
    causal = []
    contradiction = []
    boosted, suppressed = cross_reference_evidence(signals, credibility, causal, contradiction)
    assert isinstance(boosted, list)
    assert isinstance(suppressed, list)
    print("  PASS: cross_reference_evidence")


def test_apply_xref_adjustments():
    from neurons.evidence_pipeline import apply_xref_adjustments
    signals = [
        Signal(data={"evidence_text": "test"}, signal_type=SignalType.EVIDENCE, confidence=0.8, source_center="test"),
    ]
    boosted = {"0": 0.1}
    suppressed = {}
    result = apply_xref_adjustments(signals, boosted, suppressed)
    assert isinstance(result, list)
    assert len(result) == 1
    print("  PASS: apply_xref_adjustments")


# ══════════════════════════════════════════════════════════════
# complexity.py
# ══════════════════════════════════════════════════════════════

def test_classify_query_complexity():
    from neurons.complexity import classify_query_complexity
    c1 = classify_query_complexity("What is 2+2?", 0)
    c2 = classify_query_complexity("Explain the relationship between quantum mechanics and general relativity", 10)
    c3 = classify_query_complexity("Hi", 0)
    assert c1 in ("trivial", "simple", "moderate", "complex", "deep")
    assert c2 in ("trivial", "simple", "moderate", "complex", "deep")
    print(f"  PASS: classify_query_complexity (simple='{c1}', complex='{c2}', trivial='{c3}')")


def test_select_reasoning_modules():
    from neurons.complexity import select_reasoning_modules
    modules = select_reasoning_modules("simple", 1)
    assert isinstance(modules, list)
    assert len(modules) > 0
    modules2 = select_reasoning_modules("complex", 10)
    assert len(modules2) >= len(modules)
    print(f"  PASS: select_reasoning_modules (simple={len(modules)}, complex={len(modules2)})")


# ══════════════════════════════════════════════════════════════
# human_reasoning.py
# ══════════════════════════════════════════════════════════════

def test_run_human_reasoning():
    from neurons.human_reasoning import run_human_reasoning
    from neurons.brain import Forebrain
    fb = Forebrain()
    result = run_human_reasoning(
        modules=["common_sense", "abductive"],
        query="Is water wet?",
        evidence_texts=["Water makes things wet"],
        sources=["test"],
        final_confidence=0.8,
        consensus_decision="supported",
        forebrain=fb,
    )
    assert hasattr(result, "common_sense_plausibility")
    assert hasattr(result, "abductive_hypotheses")
    print("  PASS: run_human_reasoning")


# ══════════════════════════════════════════════════════════════
# web_scraper: ContentExtractor
# ══════════════════════════════════════════════════════════════

def test_content_extractor():
    from neurons.web_scraper.content import ContentExtractor
    ext = ContentExtractor()
    html = '<html><head><title>Test</title><meta name="description" content="A test"></head><body><h1>Hello</h1><p>World</p><nav>X</nav></body></html>'
    result = ext.extract_from_html(html, "https://example.com")
    assert result["title"] == "Test"
    assert "Hello" in result["text"]
    assert "X" not in result["text"]  # nav should be stripped
    assert result["metadata"]["description"] == "A test"
    print("  PASS: ContentExtractor")


def test_wikipedia_cleaning():
    from neurons.web_scraper.content import ContentExtractor
    ext = ContentExtractor()
    cleaned = ext.clean_wikipedia_extract("Section:\nFact [1] [2].\nMore.")
    assert "Section" not in cleaned
    assert "[1]" not in cleaned
    print("  PASS: Wikipedia cleaning")


def test_summary_extraction():
    from neurons.web_scraper.content import ContentExtractor
    ext = ContentExtractor()
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    summary = ext.extract_summary(text, max_sentences=2)
    assert "First" in summary
    assert "Second" in summary
    print("  PASS: Summary extraction")


# ══════════════════════════════════════════════════════════════
# web_scraper: WebScraper
# ══════════════════════════════════════════════════════════════

def test_web_scraper_init():
    from neurons.web_scraper.scraper import WebScraper
    s = WebScraper()
    assert s._client is not None
    stats = s.get_stats()
    assert "total_fetches" in stats
    s.close()
    print("  PASS: WebScraper init")


def test_web_scraper_backoff():
    from neurons.web_scraper.scraper import WebScraper
    s = WebScraper()
    delays = [s._backoff_delay(i) for i in range(5)]
    assert all(d <= 10.0 for d in delays)
    assert delays[0] < delays[3]
    s.close()
    print("  PASS: Backoff delay")


def test_web_scraper_wikipedia():
    from neurons.web_scraper.scraper import WebScraper
    s = WebScraper()
    page = s.fetch("https://en.wikipedia.org/wiki/Mars")
    if page.success:
        assert page.title
        assert len(page.text) > 50
        assert page.source == "wikipedia"
    s.close()
    print("  PASS: Wikipedia fetch")


def test_web_scraper_duckduckgo():
    from neurons.web_scraper.scraper import WebScraper
    s = WebScraper()
    ddg = s._search_duckduckgo("Python programming")
    if ddg and ddg.success:
        assert ddg.title
        assert len(ddg.text) > 10
    s.close()
    print("  PASS: DuckDuckGo search")


def test_web_scraper_cache(monkeypatch):
    from neurons.web_scraper.scraper import WebScraper, ScrapedPage
    s = WebScraper(cache_size=100)
    calls = []

    def fetch(url):
        calls.append(url)
        return ScrapedPage(url=url, title="Mars", text="Mars is a planet.")

    monkeypatch.setattr(s, "_fetch_wikipedia", fetch)
    p1 = s.fetch("https://en.wikipedia.org/wiki/Mars")
    p2 = s.fetch("https://en.wikipedia.org/wiki/Mars")
    stats = s.get_stats()
    assert stats["cache_hits"] >= 1
    assert len(calls) == 1 and p1.text == p2.text
    s.close()
    print("  PASS: Cache hit")


# ══════════════════════════════════════════════════════════════
# web_scraper: PDFScraper
# ══════════════════════════════════════════════════════════════

def test_pdf_scraper_init():
    from neurons.web_scraper.pdf_scraper import PDFScraper
    pdf = PDFScraper()
    stats = pdf.get_stats()
    assert "total_extractions" in stats
    pdf.close()
    print("  PASS: PDFScraper init")


def test_pdf_backoff():
    from neurons.web_scraper.pdf_scraper import PDFScraper
    pdf = PDFScraper()
    delays = [pdf._backoff_delay(i) for i in range(4)]
    assert delays[0] < delays[2]
    pdf.close()
    print("  PASS: PDF backoff")


def test_pdf_from_bytes():
    from neurons.web_scraper.pdf_scraper import PDFScraper
    pdf = PDFScraper()
    minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 24 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000360 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
435
%%EOF"""
    result = pdf.extract_from_bytes(minimal_pdf)
    # pypdf should handle this
    pdf.close()
    print(f"  PASS: PDF from bytes (success={result.success})")


# ══════════════════════════════════════════════════════════════
# web_scraper: HeadlessBrowser
# ══════════════════════════════════════════════════════════════

def test_headless_browser_init():
    from neurons.web_scraper.headless_browser import HeadlessBrowser
    b = HeadlessBrowser()
    renderer = b.get_renderer()
    assert renderer in ("playwright", "selenium", "none", "http_fallback")
    stats = b.get_stats()
    assert "total_renders" in stats
    b.close()
    print("  PASS: HeadlessBrowser init")


def test_headless_browser_render():
    from neurons.web_scraper.headless_browser import HeadlessBrowser
    b = HeadlessBrowser()
    page = b.render("https://en.wikipedia.org/wiki/Python_(programming_language)", wait_ms=500)
    # Should succeed even without a real browser (HTTP fallback)
    assert page.url
    b.close()
    print(f"  PASS: HeadlessBrowser render (renderer={page.renderer})")


# ══════════════════════════════════════════════════════════════
# web_scraper: WebResearcher
# ══════════════════════════════════════════════════════════════

def test_researcher_query_generation():
    from neurons.web_scraper.researcher import WebResearcher
    from neurons.web_scraper.scraper import WebScraper
    r = WebResearcher(WebScraper())
    queries = r._generate_queries("machine learning")
    assert len(queries) >= 4
    assert any("machine learning" in q.lower() for q in queries)
    print(f"  PASS: Query generation ({len(queries)} queries)")


def test_researcher_question_queries():
    from neurons.web_scraper.researcher import WebResearcher
    from neurons.web_scraper.scraper import WebScraper
    r = WebResearcher(WebScraper())
    queries = r._generate_question_queries("Who invented the telephone?")
    assert len(queries) >= 2
    assert any("telephone" in q.lower() for q in queries)
    print(f"  PASS: Question queries ({len(queries)} queries)")


def test_researcher_entity_extraction():
    from neurons.web_scraper.researcher import WebResearcher
    from neurons.web_scraper.scraper import WebScraper
    r = WebResearcher(WebScraper())
    entities = r._extract_entities("Albert Einstein worked at Princeton University.")
    assert len(entities) >= 1
    assert any("Albert" in e or "Einstein" in e for e in entities)
    print(f"  PASS: Entity extraction ({len(entities)} entities)")


def test_researcher_key_facts():
    from neurons.web_scraper.researcher import WebResearcher
    from neurons.web_scraper.scraper import WebScraper
    r = WebResearcher(WebScraper())
    facts = r._extract_key_facts("Einstein was born in 1879. The tower is 330 meters tall.")
    assert len(facts) >= 1
    print(f"  PASS: Key fact extraction ({len(facts)} facts)")


# ══════════════════════════════════════════════════════════════
# CORE PROTOCOL
# ══════════════════════════════════════════════════════════════

def test_core_protocol():
    from neurons.core_protocol import CoreResult
    result = CoreResult(
        core_id="test", answer="yes", confidence=0.9,
        reasoning="because", latency_ms=1.0,
    )
    assert result.core_id == "test"
    assert result.answer == "yes"
    print("  PASS: CoreResult")


# ══════════════════════════════════════════════════════════════
# EVOLUTION
# ══════════════════════════════════════════════════════════════

def test_learning_module():
    from neurons.evolution.learning import LearningModule, LearningEvent
    lm = LearningModule()
    event = LearningEvent(
        query="test", expected="yes", actual="yes",
        correct=True, confidence=0.8, source="test",
    )
    lm.record_event(event)
    stats = lm.get_stats()
    assert stats.get("total_events", 0) >= 1
    print("  PASS: LearningModule")


def test_evolution_engine():
    from neurons.evolution.engine import EvolutionEngine
    ee = EvolutionEngine()
    # Test mutation (failure_count must be int)
    mutation = ee.mutate_pattern("what is *", failure_count=3, context="test")
    assert mutation.original == "what is *"
    assert mutation.mutated
    # Test cross-pollination
    child = ee.cross_pollinate("pattern A", "pattern B")
    assert isinstance(child, str)
    # Test fitness recording
    ee.record_fitness(0.8)
    print("  PASS: EvolutionEngine")


def test_knowledge_acquisition():
    from neurons.evolution.knowledge import KnowledgeAcquisition
    ka = KnowledgeAcquisition()
    # Test acquire
    ka.acquire("Water is H2O", "science", 0.9)
    # Test get_knowledge
    knowledge = ka.get_knowledge(min_confidence=0.5)
    assert isinstance(knowledge, list)
    # Test stats
    stats = ka.get_stats()
    assert isinstance(stats, dict)
    print(f"  PASS: KnowledgeAcquisition (knowledge={len(knowledge)}, stats={stats})")


# ══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("UNIT TESTS — REFACTORED MODULES + WEB SCRAPER")
    print("=" * 60)

    tests = [
        # trace.py
        test_reasoning_trace,
        test_reasoning_result,
        # fast_path.py
        test_fast_path,
        test_quick_direction,
        # evidence_pipeline.py
        test_cross_reference_evidence,
        test_apply_xref_adjustments,
        # complexity.py
        test_classify_query_complexity,
        test_select_reasoning_modules,
        # human_reasoning.py
        test_run_human_reasoning,
        # ContentExtractor
        test_content_extractor,
        test_wikipedia_cleaning,
        test_summary_extraction,
        # WebScraper
        test_web_scraper_init,
        test_web_scraper_backoff,
        test_web_scraper_wikipedia,
        test_web_scraper_duckduckgo,
        test_web_scraper_cache,
        # PDFScraper
        test_pdf_scraper_init,
        test_pdf_backoff,
        test_pdf_from_bytes,
        # HeadlessBrowser
        test_headless_browser_init,
        test_headless_browser_render,
        # WebResearcher
        test_researcher_query_generation,
        test_researcher_question_queries,
        test_researcher_entity_extraction,
        test_researcher_key_facts,
        # Core protocol
        test_core_protocol,
        # Evolution
        test_learning_module,
        test_evolution_engine,
        test_knowledge_acquisition,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__} - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
