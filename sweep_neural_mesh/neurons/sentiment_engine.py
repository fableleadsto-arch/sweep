"""
Sentiment Analysis — keyword-scored with sklearn fallback and optional transformer.

Architecture:
    Text Input
        ↓
    [Primary: keyword scoring with sentiment lexicon]
    [Fallback: TF-IDF + MultinomialNB (fast train)]
    [Optional: distilbert transformer (slow load)]
        ↓
    Sentiment (positive/negative/neutral) + score (0.0–1.0)
        ↓
    Valence: -1.0 (negative) to +1.0 (positive)

Keyword scoring loads instantly, no model downloads needed.
"""
from __future__ import annotations

from services.model_loading import serialized_model_load

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    text: str
    label: SentimentLabel
    score: float
    valence: float          # -1.0 (very negative) to +1.0 (very positive)
    confidence: float       # model confidence in its own prediction
    backend: str
    latency_ms: float = 0.0
    raw_scores: dict[str, float] = field(default_factory=dict)


# ── Sentiment Lexicon (built-in, no downloads) ──
_POSITIVE_WORDS = {
    "great", "good", "excellent", "amazing", "wonderful", "fantastic",
    "outstanding", "brilliant", "superb", "perfect", "beautiful", "happy",
    "love", "glad", "thrilled", "impressive", "exceptional", "delightful",
    "pleasant", "awesome", "best", "nice", "like", "enjoy", "pleased",
    "thankful", "grateful", "proud", "excited", "elegant", "elegant",
    "helpful", "useful", "effective", "efficient", "smart", "fast",
    "strong", "powerful", "innovative", "successful", "victory", "win",
    "success", "progress", "improve", "improved", "improvement", "positive",
    "right", "correct", "true", "agree", "support", "confirm", "verified",
    "solved", "fixed", "resolved", "complete", "done", "finished",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "disgusting", "dreadful",
    "worst", "hate", "sad", "angry", "frustrated", "annoying", "ugly",
    "broken", "useless", "poor", "wrong", "false", "fail", "failure",
    "error", "bug", "crash", "problem", "issue", "confused", "lost",
    "slow", "weak", "stupid", "waste", "spam", "scam", "fraud", "danger",
    "risk", "threat", "attack", "virus", "malware", "crash", "lose",
    "loss", "dead", "death", "kill", "destroy", "damage", "harm",
    "pain", "suffer", "depressed", "lonely", "fear", "scared", "afraid",
    "anxious", "worried", "trouble", "difficult", "hard", "impossible",
}

_INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "incredibly": 2.0, "absolutely": 2.0,
    "really": 1.5, "truly": 1.5, "totally": 1.5, "completely": 1.8,
    "highly": 1.5, "remarkably": 1.5, "exceptionally": 2.0,
}

_NEGATORS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "can't", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "isn't", "aren't", "wasn't", "weren't", "hardly", "barely",
}


def _keyword_sentiment(text: str) -> tuple[float, dict[str, float]]:
    """
    Keyword-based sentiment scoring using lexicon lookup.

    Returns (valence, raw_scores) where valence is -1.0 to +1.0.
    """
    words = re.findall(r'\b[a-z\']+\b', text.lower())
    if not words:
        return 0.0, {"positive": 0.0, "negative": 0.0}

    pos_score = 0.0
    neg_score = 0.0
    negated = False
    intensifier = 1.0

    for i, word in enumerate(words):
        if word in _NEGATORS:
            negated = True
            continue
        if word in _INTENSIFIERS:
            intensifier = _INTENSIFIERS[word]
            continue

        if word in _POSITIVE_WORDS:
            if negated:
                neg_score += 0.8 * intensifier
            else:
                pos_score += 1.0 * intensifier
        elif word in _NEGATIVE_WORDS:
            if negated:
                pos_score += 0.5 * intensifier
            else:
                neg_score += 1.0 * intensifier

        negated = False
        intensifier = 1.0

    total = pos_score + neg_score
    if total == 0:
        valence = 0.0
        raw = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    else:
        valence = (pos_score - neg_score) / total
        raw = {
            "positive": pos_score / max(1, total),
            "negative": neg_score / max(1, total),
            "neutral": 0.0,
        }

    return max(-1.0, min(1.0, valence)), raw


class SentimentEngine:
    """Sentiment analysis.

    Backend: pretrained distilbert-sst2 (HF: distilbert-base-uncased-
    finetuned-sst-2-english, 268MB, 96%+ SST-2 accuracy) loaded lazily in a
    background thread; the built-in keyword scorer answers instantly until
    the neural backend is ready. Cortex interface unchanged.
    """

    # HF id and local cache path are resolved in the loader thread
    HF_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        self._backend = "keyword"
        self._nn_model = None
        self._nn_tok = None
        self._nn_state = None  # None=idle, "loading", True=ready, False=failed
        self._nn_lock = threading.Lock()
        self._start_nn_load()

    # ── neural backend (lazy, non-blocking) ───────────────────────────
    def _start_nn_load(self) -> None:
        if self._nn_state is None:
            self._nn_state = "loading"
            threading.Thread(target=self._nn_load_worker, daemon=True).start()

    @serialized_model_load
    def _nn_load_worker(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.HF_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(self.HF_MODEL)
            model.eval()
            torch.set_num_threads(min(4, os.cpu_count() or 2))
            with self._nn_lock:
                self._nn_tok, self._nn_model = tok, model
                self._nn_state = True
                self._backend = "distilbert-sst2"
        except Exception as e:
            logger.warning("distilbert-sst2 load failed, staying on keyword: %s", e)
            self._nn_state = False

    def _nn_ready(self) -> bool:
        return self._nn_state is True and self._nn_model is not None

    def _nn_analyze(self, text: str) -> tuple[float, dict] | None:
        if not self._nn_ready():
            return None
        try:
            import torch
            inputs = self._nn_tok(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = self._nn_model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0)
            neg, pos = float(probs[0]), float(probs[1])  # sst2: 0=neg, 1=pos
            valence = pos - neg
            raw = {"positive": round(pos, 4), "negative": round(neg, 4),
                   "neutral": round(1.0 - max(pos, neg), 4)}
            return valence, raw
        except Exception as e:
            logger.debug("nn sentiment failed: %s", e)
            return None

    def analyze(self, text: str) -> SentimentResult:
        t0 = time.perf_counter()

        nn = self._nn_analyze(text)
        if nn is not None:
            valence, raw = nn
        else:
            valence, raw = _keyword_sentiment(text)
        confidence = 1.0 - raw.get("neutral", 0.0)

        if abs(valence) < 0.15:
            label = SentimentLabel.NEUTRAL
        elif valence > 0:
            label = SentimentLabel.POSITIVE
        else:
            label = SentimentLabel.NEGATIVE

        return SentimentResult(
            text=text, label=label, score=confidence,
            valence=valence, confidence=confidence,
            backend=self._backend if nn is not None else "keyword",
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw_scores=raw,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]

    def sentiment_score(self, text: str) -> float:
        return self.analyze(text).valence


_default_sentiment: SentimentEngine | None = None


def get_sentiment_engine() -> SentimentEngine:
    global _default_sentiment
    if _default_sentiment is None:
        _default_sentiment = SentimentEngine()
    return _default_sentiment
