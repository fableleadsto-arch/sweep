"""
General Knowledge Engine — LLM-competitive general knowledge answering.

Pipeline (fastest → slowest):
  1. Mega KB lookup (case-insensitive + fuzzy match, ~0.1ms)
  2. KnowledgeTrainer + SupplementaryKnowledge lookup
  3. Semantic retrieval over all KB entries (SentenceTransformer)
  4. Wikipedia/Wikidata RAG with extractive QA answer extraction
  5. Local generative LLM fallback (open-ended questions)

Usage:
    from sweep_neural_mesh.neurons.general_knowledge import GeneralKnowledge
    gk = GeneralKnowledge()
    result = gk.answer("What is the capital of France?")
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .knowledge_mega import get_mega_knowledge


@dataclass
class KnowledgeAnswer:
    """Result from the general knowledge engine."""
    answer: str
    confidence: float
    method: str
    source: str = ""
    latency_ms: float = 0.0
    reasoning: str = ""
    facts_used: list[str] = field(default_factory=list)
    targeted: bool = True
    open_ended: bool = False


class GeneralKnowledge:
    """Answer general-knowledge questions with multiple fallback tiers."""

    def __init__(self, enable_live: bool = True, enable_llm: bool = True) -> None:
        self._enable_live = enable_live
        self._enable_llm = enable_llm
        self._mega = get_mega_knowledge()
        self._entries: list = self._mega.get_all()

        # Supplementary tiers
        self._kb_entries: list = []
        self._supp_entries: list = []
        try:
            from .knowledge_training import KnowledgeTrainer
            self._kb_entries = KnowledgeTrainer().get_all()
        except Exception:
            pass
        try:
            from .knowledge_supplement import SupplementaryKnowledge
            self._supp_entries = SupplementaryKnowledge().get_all()
        except Exception:
            pass

        # Lazy-loaded heavy components
        self._embedder = None
        self._live = None
        self._llm = None
        self._llm_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._loaded_light = False

        self._stats = {"kb_hits": 0, "semantic_hits": 0, "live_hits": 0, "llm_hits": 0, "misses": 0}

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def answer(self, query: str, timeout_live: float = 4.0) -> KnowledgeAnswer:
        """Answer a general knowledge question."""
        t0 = time.perf_counter()
        q = query.strip().rstrip("?.!")

        # Tier 0.5: Superlative lookup ("largest X" should match a
        # "largest planet"-style topic before a generic "solar system" one)
        result = self._lookup_superlative(q)
        if result is not None:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Tier 1: Mega KB exact + fuzzy
        result = self._lookup_mega(q)
        if result is not None:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Tier 2: KnowledgeTrainer + Supplementary
        result = self._lookup_kb(q)
        if result is not None:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Detect open-ended / non-factoid questions (lists, "name X",
        # explanations, opinions) — the LLM handles these better than
        # extractive RAG.
        open_ended = bool(re.search(
            r"\b(?:name|list|explain|describe|compare|difference between|how do|how does|why is|why do|why does|what is the difference|tell me|write|give me|what are the main)\b",
            q.lower(),
        ))

        # Tier 3: Live RAG (Wikipedia + extractive QA) — cheap and reliable.
        # For open-ended questions, a generic first-sentence extraction is
        # weak, so if an LLM is available let it answer instead.
        live_result = None
        if self._enable_live:
            live_result = self._lookup_live(q, timeout_live)
        if live_result is not None and live_result.confidence >= 0.85:
            live_result.latency_ms = (time.perf_counter() - t0) * 1000
            return live_result

        # Tier 4: Local generative LLM (preferred for open-ended/weak cases)
        if self._enable_llm and (open_ended or live_result is None or live_result.confidence < 0.8):
            llm_result = self._lookup_llm(q)
            if llm_result is not None:
                llm_result.latency_ms = (time.perf_counter() - t0) * 1000
                return llm_result

        # Tier 5: Accept a weaker live extraction if the LLM was unavailable
        if live_result is not None:
            live_result.latency_ms = (time.perf_counter() - t0) * 1000
            return live_result

        # Tier 6: Semantic retrieval (slow to load, so only reached on live miss)
        result = self._lookup_semantic(q)
        if result is not None:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        self._stats["misses"] += 1
        return KnowledgeAnswer(
            answer="I don't have enough information to answer that.",
            confidence=0.1, method="unknown",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    # ══════════════════════════════════════════════════════════
    # TIER 0.5: SUPERLATIVES
    # ══════════════════════════════════════════════════════════

    _SUPERLATIVES = ("largest", "biggest", "tallest", "highest", "smallest", "longest", "deepest", "fastest", "oldest", "most")

    def _lookup_superlative(self, query: str) -> KnowledgeAnswer | None:
        m = re.search(
            r"\b(?:what|which)\s+(?:is|are)\s+the\s+(largest|biggest|tallest|highest|smallest|longest|deepest|fastest|oldest|most)\s+([a-z]+)",
            query.lower(),
        )
        if not m:
            return None
        sup, noun = m.group(1), m.group(2).strip()
        candidates = [f"{sup} {noun}", noun]
        for e in self._entries:
            t = self._normalize(e.topic)
            for cand in candidates:
                if t == self._normalize(cand):
                    self._stats["kb_hits"] += 1
                    return KnowledgeAnswer(
                        answer=e.answer, confidence=e.confidence, method="kb",
                        source=e.source, reasoning=e.fact, facts_used=[e.topic],
                    )
        return None

    # ══════════════════════════════════════════════════════════
    # TIER 1: MEGA KB
    # ══════════════════════════════════════════════════════════

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()

    _FILLER_WORDS = {"is", "are", "was", "were", "be", "been", "the", "a", "an", "of", "in", "on", "at", "to", "for", "that", "which", "what", "who", "when", "where", "how", "do", "does", "did", "have", "has", "as", "known", "by", "and", "or", "with", "from", "called", "known as", "also", "its", "it", "their"}

    @staticmethod
    def _expand_aliases(text: str) -> str:
        """Normalise country-name abbreviations so "UK" matches "United Kingdom"."""
        t = text
        t = re.sub(r"\buk\b", "united kingdom", t)
        t = re.sub(r"\bu\.?k\.?\b", "united kingdom", t)
        t = re.sub(r"\bus\b", "united states", t)
        t = re.sub(r"\bu\.?s\.?a?\.?\b", "united states", t)
        return t

    @staticmethod
    def _stem_light(w: str) -> str:
        """Minimal suffix stemming so "landed"/"landing" match "land"."""
        if len(w) <= 4:
            return w
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                base = w[: -len(suf)]
                # avoid "is" -> "" style accidents and keep real words
                if len(base) >= 3:
                    return base
        return w

    @staticmethod
    def _fuzzy_match(q_words: list[str], t_words: list[str]) -> bool:
        """Match with filler words ignored on both sides.

        Tries an ordered subsequence first (``"capital of france"`` vs
        ``"what is the capital of france"``); if the word order differs
        (``"which country has the largest population"`` vs ``"largest
        country by population"``), falls back to an unordered subset check.
        Light stemming makes "landed" match "land".
        """
        if len(t_words) < 2:
            return False
        from collections import Counter
        q_stem = [GeneralKnowledge._stem_light(w) for w in q_words]
        t_stem = [GeneralKnowledge._stem_light(w) for w in t_words]
        # Ordered subsequence (same word order)
        it = iter(q_stem)
        if all(w in it for w in t_stem):
            return True
        # Unordered: every content word of the topic must appear in the query
        qc = Counter(q_stem)
        tc = Counter(t_stem)
        if all(qc[w] >= tc[w] for w in tc):
            # Require reasonable overlap so we don't match 2 shared words of a 6-word topic
            return len(t_words) >= 2 and len(tc) / max(len(qc), 1) >= 0.35
        return False

    def _lookup_mega(self, query: str) -> KnowledgeAnswer | None:
        q = self._normalize(query)

        # "how many X ..." -> "number of X ..." so topics like
        # "number of planets in solar system" match
        if q.startswith("how many "):
            rest = q[len("how many "):]
            # Drop filler verbs only: "how many planets are in" -> "planets in"
            rest = re.sub(r"\b(?:are|is|do|does|span|spans|cover|covers|make up|exist)\b", " ", rest)
            rest = re.sub(r"\s+", " ", rest).strip()
            alts = [f"number of {rest}", rest]
            for e in self._entries:
                t = self._normalize(e.topic)
                for alt in alts:
                    if t == alt:
                        self._stats["kb_hits"] += 1
                        return KnowledgeAnswer(
                            answer=e.answer, confidence=e.confidence, method="kb",
                            source=e.source, reasoning=e.fact,
                            facts_used=[e.topic],
                        )
                # Also try with/without a "the" between words
                alt_norm = self._normalize(alt)
                t_norm = self._normalize(t)
                if re.search(rf"\b{re.escape(t_norm)}\b", alt_norm.replace(" the ", " ")
                             .replace(" in ", " ").replace(" of ", " ")):
                    self._stats["kb_hits"] += 1
                    return KnowledgeAnswer(
                        answer=e.answer, confidence=e.confidence, method="kb",
                        source=e.source, reasoning=e.fact,
                        facts_used=[e.topic],
                    )
                # "span"-type: topic "number of time zones in russia" vs
                # alt "time zones russia" — check the noun phrase matches
                t_core = re.sub(r"^(?:number|total)\s+of\s+", "", t_norm)
                t_core = re.sub(r"\b(?:in|on|of|for|the|a|an)\b", " ", t_core)
                t_core = re.sub(r"\s+", " ", t_core).strip()
                alt_core = re.sub(r"\b(?:in|on|of|for|the|a|an)\b", " ", alt_norm)
                alt_core = re.sub(r"\s+", " ", alt_core).strip()
                if t_core and t_core == alt_core:
                    self._stats["kb_hits"] += 1
                    return KnowledgeAnswer(
                        answer=e.answer, confidence=e.confidence * 0.95, method="kb",
                        source=e.source, reasoning=e.fact,
                        facts_used=[e.topic],
                    )

        # Try exact topic match first (case-insensitive, alias-aware)
        for e in self._entries:
            t = self._expand_aliases(self._normalize(e.topic))
            if q == t:
                self._stats["kb_hits"] += 1
                return KnowledgeAnswer(
                    answer=e.answer, confidence=e.confidence, method="kb",
                    source=e.source, reasoning=e.fact,
                    facts_used=[e.topic],
                )

        q_exp = self._expand_aliases(q)
        q_words = [w for w in q_exp.split() if w not in self._FILLER_WORDS]

        # Pass 1: exact substring containment (word-boundary aware, so
        # "world war i" doesn't match inside "world war ii")
        best = None
        best_score = 0.0
        for e in self._entries:
            t = self._normalize(e.topic)
            if not t:
                continue
            in_q = re.search(rf"\b{re.escape(t)}\b", q_exp) is not None
            if in_q or (len(q) >= 3 and re.search(rf"\b{re.escape(q)}\b", t)):
                score = len(t) / max(len(q), 1)
                if score > best_score:
                    best_score = score
                    best = e
        if best is not None and best_score >= 0.55:
            self._stats["kb_hits"] += 1
            return KnowledgeAnswer(
                answer=best.answer, confidence=best.confidence * 0.95, method="kb",
                source=best.source, reasoning=best.fact,
                facts_used=[best.topic],
            )

        # Pass 2: fuzzy ordered-subsequence matching so minor filler words
        # don't hide a fact (e.g. "country IS known as" vs "country known as")
        best2 = None
        best2_len = 0
        for e in self._entries:
            t = self._expand_aliases(self._normalize(e.topic))
            if not t or len(t) < 6:
                continue
            t_words = [w for w in t.split() if w not in self._FILLER_WORDS]
            if len(t_words) < 2:
                continue
            if self._fuzzy_match(q_words, t_words):
                # Longer topic = more specific; prefer a high word-overlap
                overlap = len(t_words) / max(len(set(t_words)) + 1, 1)
                if len(t) > best2_len and overlap >= 0.6:
                    best2_len = len(t)
                    best2 = e
        if best2 is not None and best2_len >= 8:
            self._stats["kb_hits"] += 1
            return KnowledgeAnswer(
                answer=best2.answer, confidence=best2.confidence * 0.9, method="kb",
                source=best2.source, reasoning=best2.fact,
                facts_used=[best2.topic],
            )
        return None

    # ══════════════════════════════════════════════════════════
    # TIER 2: KNOWLEDGE TRAINER + SUPPLEMENT
    # ══════════════════════════════════════════════════════════

    def _lookup_kb(self, query: str) -> KnowledgeAnswer | None:
        q = self._normalize(query)
        all_entries = list(self._kb_entries) + list(self._supp_entries)
        if not all_entries:
            return None

        # Prefer the longest matching topic (most specific), so e.g.
        # "number of planets in solar system" beats "solar system". Entries
        # whose answer is a bare yes/no are existence facts, not answers to
        # "what is the currency of X"-style questions — skip them when a
        # substantive match exists.
        _TRIVIAL = {"yes", "no", "true", "false", "unknown", "n/a", "-", ""}
        best = None
        best_len = -1
        trivial_best = None
        trivial_len = -1
        for e in all_entries:
            topic = getattr(e, "topic", "") or ""
            t = self._normalize(topic)
            if not t:
                continue
            if q == t or (len(t) >= 5 and re.search(rf"\b{re.escape(t)}\b", q)):
                answer = str(getattr(e, "answer", "")).strip().lower()
                if answer in _TRIVIAL:
                    if len(t) > trivial_len:
                        trivial_len = len(t)
                        trivial_best = e
                elif len(t) > best_len:
                    best_len = len(t)
                    best = e
        # A direct full-query match wins even if the answer looks trivial
        # (it is what the KB actually says); otherwise prefer substantive.
        chosen = best if best is not None else (trivial_best if q == self._normalize(getattr(trivial_best, "topic", "")) else None)
        if chosen is not None:
            topic = getattr(chosen, "topic", "") or ""
            self._stats["kb_hits"] += 1
            conf = getattr(chosen, "confidence", 0.9) or 0.9
            return KnowledgeAnswer(
                answer=getattr(chosen, "answer", ""), confidence=conf, method="kb",
                source=getattr(chosen, "source", "knowledge_base"),
                reasoning=getattr(chosen, "fact", ""), facts_used=[topic],
            )
        return None

    # ══════════════════════════════════════════════════════════
    # TIER 3: SEMANTIC RETRIEVAL
    # ══════════════════════════════════════════════════════════

    def _load_embedder(self) -> bool:
        if self._embedder is not None:
            return True
        with self._load_lock:
            if self._embedder is not None:
                return True
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                return True
            except Exception:
                self._embedder = False  # type: ignore[assignment]
                return False

    def _lookup_semantic(self, query: str) -> KnowledgeAnswer | None:
        if not self._load_embedder():
            return None
        if not isinstance(self._embedder, object) or self._embedder is False:
            return None

        try:
            import os
            import numpy as np
            t0 = time.perf_counter()
            q_emb = self._embedder.encode(query, normalize_embeddings=True)
            # Build topic list lazily, caching embeddings to disk
            if not hasattr(self, "_topic_cache"):
                topics = [e.topic for e in self._entries]
                self._topic_cache = topics
                cache_path = os.path.join(
                    os.path.dirname(__file__), "..", "training", "neural_models", "topic_embs.npy"
                )
                cache_path = os.path.abspath(cache_path)
                try:
                    if os.path.exists(cache_path):
                        self._topic_embs = np.load(cache_path)
                    else:
                        self._topic_embs = self._embedder.encode(topics, normalize_embeddings=True, batch_size=64)
                        np.save(cache_path, self._topic_embs)
                except Exception:
                    self._topic_embs = self._embedder.encode(topics, normalize_embeddings=True, batch_size=64)
            scores = self._topic_embs @ q_emb
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            if score >= 0.72:
                self._stats["semantic_hits"] += 1
                e = self._entries[idx]
                return KnowledgeAnswer(
                    answer=e.answer, confidence=min(0.95, score), method="semantic",
                    source=e.source, reasoning=e.fact, facts_used=[e.topic],
                )
        except Exception:
            pass
        return None

    # ══════════════════════════════════════════════════════════
    # TIER 4: LIVE RAG (WIKIPEDIA + EXTRACTIVE QA)
    # ══════════════════════════════════════════════════════════

    def _extract_topic(self, query: str) -> str:
        """Extract the entity topic from a question (case-insensitive on structure, case-sensitive on entity)."""
        q = query.strip()
        # "What is the capital of X?" / "Who is the president of X?" -> X
        m = re.search(r"\b(?i:capital|largest city|currency|population|language|president|leader)\s+of\s+(?:the\s+)?([A-Z][a-zA-Z .'-]+)", q)
        if m:
            return self._clean_topic(m.group(1))
        # "What is the largest X in Y?" -> Y
        m = re.search(r"\b(?i:what)\s+(?i:is|are)\s+the\s+(?i:largest|biggest|tallest|highest|smallest|deepest|longest|fastest|most|first|oldest)\s+[a-z .'-]*?\s+(?i:in|on|of)\s+(?:the\s+)?([A-Za-z][a-zA-Z .'-]+)", q)
        if m:
            return self._clean_topic(m.group(1))
        # "Who is X?" / "Who was X?" / "What is X?"
        m = re.search(r"\b(?i:who|what)\s+(?i:is|are|was|were)\s+(?:the\s+)?([A-Z][a-zA-Z .'-]+)", q)
        if m:
            return self._clean_topic(m.group(1))
        # "When was X ..." / "When did X ..." / "Where is X ..."
        m = re.search(r"\b(?i:when|where|how|why)\s+(?i:was|were|did|is|are)\s+(?:the\s+)?([A-Z][a-zA-Z .'-]+)", q)
        if m:
            return self._clean_topic(m.group(1))
        # "Who invented/discovered/founded X?" -> X (lowercase entities allowed)
        m = re.search(r"\b(?i:who)\s+(?i:invented|created|discovered|wrote|built|designed|founded)\s+(?:the\s+)?([A-Za-z][a-zA-Z .\-']*)", q)
        if m:
            return self._clean_topic(m.group(1))
        # Proper nouns as fallback (skip question words)
        proper = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", q)
        if proper:
            filtered = [p for p in proper if p.lower() not in ("who", "what", "when", "where", "why", "how", "which", "the")]
            if filtered:
                return max(filtered, key=len)
        return q

    _STOP_WORDS = re.compile(
        r"\s+(?:is|are|was|were|has|have|had|located|founded|invented|discovered|created|built|designed|ended|end|began|started|born|died|die|known|also|but|which|that|by|in|on|at|with|from|during|between)\b",
        re.IGNORECASE,
    )

    def _clean_topic(self, t: str) -> str:
        """Trim trailing sentence words (verbs/fillers) from a captured topic."""
        t = t.strip()
        m = self._STOP_WORDS.search(t)
        if m:
            t = t[: m.start()]
        return t.strip()

    def _wiki_fetch(self, topic: str, timeout: float = 4.0) -> str | None:
        """Fetch Wikipedia extract for a topic (handles SSL issues)."""
        import json
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "titles": topic, "prop": "extracts",
            "exintro": "true", "explaintext": "true", "exchars": "2500",
            "format": "json", "exlimit": 1,
        }
        from urllib.parse import urlencode
        req = urllib.request.Request(
            f"{url}?{urlencode(params)}",
            headers={"User-Agent": "SweepNeuralEngine/1.0 (research@example.com)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if pid == "-1":
                    continue
                extract = page.get("extract", "")
                if extract:
                    return extract
        except Exception:
            pass

        # Fallback: search then fetch
        try:
            search_params = {
                "action": "query", "list": "search", "srsearch": topic,
                "format": "json", "srlimit": 1,
            }
            req2 = urllib.request.Request(
                f"{url}?{urlencode(search_params)}",
                headers={"User-Agent": "SweepNeuralEngine/1.0"},
            )
            with urllib.request.urlopen(req2, timeout=timeout, context=ctx) as resp:
                sdata = json.loads(resp.read().decode())
            results = sdata.get("query", {}).get("search", [])
            if results:
                return self._wiki_fetch(results[0]["title"], timeout)
        except Exception:
            pass
        return None

    def _answer_from_extract(self, query: str, extract: str) -> tuple[str, bool] | None:
        """Extract an answer from Wikipedia text using pattern matching + extractive QA.

        Returns ``(answer, targeted)`` where ``targeted`` is False when the
        answer is only a generic first-sentence filler (definitional) rather
        than a direct answer to the question. Callers can route weak answers
        to a generative fallback.
        """
        q = query.lower()
        sentences = re.split(r"(?<=[.!?])\s+", extract)

        # Pattern: "capital of X" -> find "X is the capital" or "capital ... is X"
        capital_m = re.search(r"\bcapital\b", q)
        if capital_m:
            for s in sentences:
                if "capital" in s.lower():
                    # "Paris is the capital" or "The capital is Paris"
                    m = re.search(r"\b([A-Z][a-zA-Z '-]+)\s+is\s+(?:the\s+)?capital", s)
                    if m:
                        return m.group(1).strip(), True
                    m = re.search(r"\bcapital\s+(?:city\s+)?is\s+([A-Z][a-zA-Z '-]+)", s)
                    if m:
                        return m.group(1).strip(), True
                    # "capital and largest city of France, is Paris"
                    m = re.search(r"(?:is|,)\s+([A-Z][a-zA-Z '-]+)\s*\.", s)
                    if m and "capital" in s.lower():
                        return m.group(1).strip(), True

        # Pattern: "population of X"
        if "population" in q:
            for s in sentences:
                if "population" in s.lower():
                    m = re.search(r"population\s+of\s+([\d.,\s]+million|[\d.,]+)", s, re.IGNORECASE)
                    if m:
                        return m.group(1).strip(), True
                    m = re.search(r"(?:population|inhabitants)[^.]*?([\d.,]+\s*(?:million|billion))", s, re.IGNORECASE)
                    if m:
                        return m.group(1).strip(), True

        # Pattern: "currency of X"
        if "currency" in q:
            for s in sentences:
                if "currency" in s.lower():
                    m = re.search(r"currency\s+is\s+(?:the\s+)?([A-Za-z '-]+)", s)
                    if m:
                        return m.group(1).strip(), True

        # Pattern: "when ... X" -> first year in extract
        if re.search(r"\b(when|year|date|how old)\b", q):
            m = re.search(r"\b(1[0-9]{3}|2[0-9]{3}|[0-9]+ (?:BC|AD))\b", extract)
            if m:
                return m.group(1), True

        # Pattern: "who is the president/leader of X" -> find the current office-holder
        if re.search(r"\bwho\b.*\b(?:president|prime minister|leader|king|queen|chancellor)\b", q):
            m = re.search(r"(?:current|46th|47th|incumbent)?\s*(?:president|prime minister|leader|chancellor|king|queen)\s+(?:of\s+[A-Za-z .'-]+\s+)?(?:is|was|has been)\s+([A-Z][a-zA-Z .'-]+)", extract, re.IGNORECASE)
            if m:
                return self._clean_topic(m.group(1)), True
            m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is\s+the\s+(?:current\s+)?(?:president|prime minister|leader|chancellor|king|queen)", extract)
            if m:
                return m.group(1).strip(), True

        # Pattern: "who ... X" -> prefer 'discovered/invented by Person' or person names
        if re.search(r"\bwho\b", q):
            # "discovered by X" / "invented by X" / "patented by X" / "granted to X" etc.
            m = re.search(
                r"(?:discovered|invented|developed|created|founded|built|patented|credited with|attributed to|written by|first described|first proposed)[^.]{0,50}?(?:by|to|with)\s+([A-Z][a-zA-Z .'-]+)",
                extract,
            )
            if m:
                return self._clean_topic(m.group(1)), True
            # "The patent was granted to X" / "patent was awarded to X"
            m = re.search(r"(?:patent|credit)[^.]{0,40}?(?:granted|awarded|given|goes)[^.]{0,20}?(?:to)\s+([A-Z][a-zA-Z .'-]+)", extract, re.IGNORECASE)
            if m:
                return self._clean_topic(m.group(1)), True
            # "PERSON was the first to ..." (e.g. "Alexander Fleming was the first to show")
            m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+was\s+the\s+first", extract)
            if m:
                return self._clean_topic(m.group(1)), True
            # "Nationality + role + PERSON" (e.g. "Scottish physician Alexander Fleming")
            m = re.search(
                r"\b(?:Scottish|American|British|German|French|Italian|English|Canadian|Australian|Dutch|Swedish|Norwegian|Danish|Russian|Japanese|Chinese|Indian|Greek|Roman|Swiss|Austrian|Belgian|Irish|Polish|Portuguese|Spanish|Turkish|Israeli|Mexican)\s+(?:physician|scientist|inventor|chemist|physicist|biologist|engineer|mathematician|astronomer|botanist|philosopher|writer|author|painter|composer|explorer|doctor|researcher|professor|surgeon)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
                extract,
                re.IGNORECASE,
            )
            if m:
                return self._clean_topic(m.group(1)), True
            # "X was a ... who ..." — pick the person at start
            m = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", extract)
            if m:
                return self._clean_topic(m.group(1)), True
            # Proper noun fallback: prefer person-looking names near the verb
            verb_pos = -1
            for kw in ("discovered", "invented", "founded", "created", "wrote", "built"):
                p = extract.lower().find(kw)
                if p != -1 and (verb_pos == -1 or p < verb_pos):
                    verb_pos = p
            window = extract[max(0, verb_pos - 100):verb_pos + 200] if verb_pos != -1 else extract
            m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", window)
            if m:
                return self._clean_topic(m.group(1)), True

        # Pattern: "how many"
        if "how many" in q:
            m = re.search(r"\b([\d,]+(?:\.[\d]+)?)\b", extract)
            if m:
                return m.group(1), True

        # Pattern: "where is X" -> location words in first sentence
        if re.search(r"\bwhere\b", q):
            for s in sentences[:2]:
                m = re.search(r"(?:in|located in|situated in|based in)\s+([A-Z][a-zA-Z '-]+)", s)
                if m:
                    return m.group(1), True

        # "what is X" -> definitional first sentence (weak; not a targeted hit)
        if re.search(r"\bwhat\s+(?:is|are)\b", q):
            if sentences:
                s = sentences[0]
                words = s.split()
                return " ".join(words[:20]) + ("." if not s.endswith(".") else ""), False

        # Generic: first sentence (weak)
        if sentences:
            s = sentences[0]
            return " ".join(s.split()[:20]) + ("." if not s.endswith(".") else ""), False

        return None

    def _wikidata_officeholder(self, topic: str, query: str) -> str | None:
        """Find the current office-holder (president / prime minister / king etc.)
        for a country/entity via Wikidata's P1308 (officeholder) property."""
        import json
        import ssl
        import urllib.request
        from urllib.parse import urlencode

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = "https://www.wikidata.org/w/api.php"

        def wb(params: dict) -> dict:
            req = urllib.request.Request(
                f"{url}?{urlencode(params)}",
                headers={"User-Agent": "SweepNeuralEngine/1.0 (research@example.com)"},
            )
            with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                return json.loads(resp.read().decode())

        try:
            # 1) Find the position entity (e.g. "president of the United States")
            # Wikidata search is literal: "prime minister of United Kingdom"
            # returns nothing, "... of the United Kingdom" works; but adding
            # "the" to "Canada" breaks it. Try both variants.
            variants = [topic]
            if not re.match(r"^(?:the|a|an)\s", topic, re.IGNORECASE):
                variants.append(f"the {topic}")

            pos_queries: list[str] = []
            if re.search(r"\bprime minister\b", query, re.IGNORECASE):
                pos_queries = [f"prime minister of {v}" for v in variants] + [f"{topic} prime minister"]
            elif re.search(r"\b(?:king|queen)\b", query, re.IGNORECASE):
                pos_queries = (
                    [f"monarch of {v}" for v in variants]
                    + [f"monarchy of {v}" for v in variants]
                    + [f"king of {v}" for v in variants]
                )
            else:
                pos_queries = [f"president of {v}" for v in variants] + [f"{topic} president"]
            pos_queries += [topic]
            eid = None
            for pos_query in pos_queries:
                data = wb({"action": "wbsearchentities", "search": pos_query, "language": "en", "format": "json", "limit": 1})
                results = data.get("search", [])
                if results:
                    eid = results[0]["id"]
                    break
            if not eid:
                return None
            # 2) Read the officeholder claim (P1308)
            data2 = wb({"action": "wbgetentities", "ids": eid, "props": "claims", "format": "json"})
            claims = data2.get("entities", {}).get(eid, {}).get("claims", {})
            for prop in ("P1308", "P6", "P35"):
                if prop in claims:
                    # Sort claims by rank: preferred (current) before normal,
                    # so the current office-holder is picked, not the first-ever.
                    rank_order = {"preferred": 0, "normal": 1, "deprecated": 2}
                    prop_claims = sorted(
                        claims[prop], key=lambda c: rank_order.get(c.get("rank", "normal"), 1)
                    )
                    holder = prop_claims[0]["mainsnak"]["datavalue"]["value"]["id"]
                    # Some entities (e.g. Q22686) return empty English labels via
                    # wbgetentities; wbsearchentities always resolves the label.
                    data3 = wb({"action": "wbsearchentities", "search": holder, "language": "en", "format": "json", "limit": 1})
                    results3 = data3.get("search", [])
                    if results3 and results3[0].get("label"):
                        return results3[0]["label"]
                    data4 = wb({"action": "wbgetentities", "ids": holder, "props": "labels", "format": "json"})
                    labels = data4.get("entities", {}).get(holder, {}).get("labels", {})
                    if "en" in labels:
                        return labels["en"]["value"]
                    for lang in labels.values():
                        return lang["value"]
        except Exception:
            pass
        return None

    def _lookup_live(self, query: str, timeout: float) -> KnowledgeAnswer | None:
        try:
            t0 = time.perf_counter()
            topic = self._extract_topic(query)
            if not topic or len(topic) < 2:
                return None
            # For "who discovered/invented/founded X", the history page often
            # names the person in its intro while the main page does not.
            is_who_q = re.search(r"\bwho\s+(?:discovered|invented|created|built|designed|founded|wrote)\b", query, re.IGNORECASE)
            # Only the *current* office-holder is on Wikidata's P1308; skip
            # historical questions ("who was the 16th president") and any that
            # ask about the first/oldest/previous holder.
            is_historical = re.search(r"\b(?:was|were)\b.*\b(?:first|second|third|\d+(?:st|nd|rd|th)|previous|last|oldest|youngest)\b", query, re.IGNORECASE)
            is_leader_q = (
                re.search(r"\bwho\b.*\b(?:president|prime minister|leader|king|queen|chancellor)\b", query, re.IGNORECASE)
                and not is_historical
            )
            if is_who_q:
                # History/Discovery pages usually name the person in the intro
                # while the main page does not.
                candidates = [
                    f"History of {topic}",
                    f"Invention of {topic}",
                    f"Discovery of {topic}",
                    topic,
                ]
            elif is_leader_q:
                # Current office-holders change frequently; ask Wikidata
                # first, then fall back to the Wikipedia office page.
                holder = None
                # Retry a few times for transient network/rate-limit hiccups
                for attempt in range(3):
                    holder = self._wikidata_officeholder(topic, query)
                    if holder:
                        break
                    time.sleep(0.4 * (attempt + 1))
                if holder:
                    self._stats["live_hits"] += 1
                    return KnowledgeAnswer(
                        answer=holder, confidence=0.88, method="wikidata",
                        source="wikidata:officeholder", facts_used=[topic],
                        reasoning=f"Current office-holder for {topic} per Wikidata",
                    )
                candidates = [f"President of {topic}", f"Prime Minister of {topic}", topic]
            else:
                candidates = [topic]
            extract = None
            used_topic = topic
            answer = None
            targeted = False
            for cand in candidates:
                extract = self._wiki_fetch(cand, timeout)
                if extract:
                    used_topic = cand
                    extracted = self._answer_from_extract(query, extract)
                    if extracted:
                        answer, targeted = extracted
                        break
            if not answer:
                return None
            # Confidence based on answer quality: targeted extractions are
            # trusted; generic first-sentence fillers are weak and should let
            # a generative fallback take over when one is available.
            conf = 0.9 if (targeted and len(answer) > 1) else (0.62 if len(answer) > 3 else 0.5)
            self._stats["live_hits"] += 1
            return KnowledgeAnswer(
                answer=answer, confidence=conf, method="live_rag",
                source=f"wikipedia:{used_topic}",
                reasoning=extract[:300],
                facts_used=[used_topic],
                targeted=targeted,
            )
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════
    # TIER 5: LOCAL GENERATIVE LLM
    # ══════════════════════════════════════════════════════════

    def _start_llm_load(self) -> None:
        """Kick off the model download+load in a background thread so that
        answer() never blocks on it (the download is hundreds of MB)."""
        with self._load_lock:
            if self._llm is not None or getattr(self, "_llm_loading", False):
                return
            self._llm_loading = True
        threading.Thread(target=self._load_llm_worker, daemon=True).start()

    def _load_llm_worker(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = "Qwen/Qwen2-0.5B-Instruct"
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            self._llm_model.eval()
            self._llm = True
        except Exception:
            self._llm = False
        finally:
            self._llm_loading = False

    def _load_llm(self) -> bool:
        # Never block on download/load: returns immediately.
        if self._llm is None and not getattr(self, "_llm_loading", False):
            self._start_llm_load()
        return self._llm is True

    def _lookup_llm(self, query: str) -> KnowledgeAnswer | None:
        if not self._enable_llm or not self._load_llm():
            return None
        try:
            import torch
            with self._llm_lock:
                messages = [
                    {"role": "system", "content": "Answer the question concisely with a short factual answer. If unsure, say you don't know."},
                    {"role": "user", "content": query},
                ]
                text = self._llm_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._llm_tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    output = self._llm_model.generate(
                        **inputs, max_new_tokens=64,
                        pad_token_id=self._llm_tokenizer.eos_token_id,
                    )
                response = self._llm_tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                if response and response.lower() not in ("i don't know", "i don't know.", "unknown"):
                    self._stats["llm_hits"] += 1
                    return KnowledgeAnswer(
                        answer=response, confidence=0.65, method="llm",
                        source="local_llm:Qwen2-0.5B-Instruct",
                    )
        except Exception:
            pass
        return None


# Singleton
_instance: GeneralKnowledge | None = None


def get_general_knowledge() -> GeneralKnowledge:
    global _instance
    if _instance is None:
        _instance = GeneralKnowledge()
    return _instance