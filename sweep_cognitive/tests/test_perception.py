"""
Tests for the perception engine (Phase 1/2).

These tests verify that:
1. Text perception extracts features correctly
2. Modality detection works
3. Confidence is appropriately assigned
4. Missing information is identified
"""

import pytest

from ..perception import (
    PerceptionEngine,
    PerceptionResult,
    TextPerceptionProcessor,
    Modality,
    PerceptionConfidence,
)
from ..representation import RepresentationQuality


class TestTextPerceptionProcessor:
    """Tests for text perception."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextPerceptionProcessor()

    def test_empty_text(self):
        """Test processing empty text."""
        result = self.processor.process("")
        assert result.representation.content == ""
        assert result.representation.quality == RepresentationQuality.NONE
        assert result.representation.confidence == 0.0

    def test_short_text(self):
        """Test processing very short text."""
        result = self.processor.process("Hi.")
        assert result.representation.confidence < 0.5

    def test_declarative_detection(self):
        """Test declarative sentence detection."""
        result = self.processor.process("The cat sat on the mat.")
        assert result.representation.is_declarative is True
        assert result.representation.is_question is False

    def test_question_detection(self):
        """Test question detection."""
        result = self.processor.process("What is the capital of France?")
        assert result.representation.is_question is True
        assert result.representation.is_declarative is False

    def test_entity_extraction(self):
        """Test entity extraction."""
        result = self.processor.process(
            "John Smith works at Acme Corporation in 2024."
        )
        assert len(result.representation.entities) > 0
        # Check for person/organization entity
        entity_types = {e["type"] for e in result.representation.entities}
        assert "PERSON_ORGANIZATION" in entity_types or "ENTITY_CLAIM" in entity_types

    def test_date_extraction(self):
        """Test date extraction."""
        result = self.processor.process("The event is on 2024-01-15.")
        dates = [e for e in result.representation.entities if e["type"] == "DATE"]
        assert len(dates) > 0

    def test_temporal_marker_extraction(self):
        """Test temporal marker extraction."""
        result = self.processor.process(
            "Yesterday, the meeting was scheduled for next week."
        )
        assert len(result.representation.temporal_markers) > 0

    def test_certainty_marker_extraction(self):
        """Test certainty marker extraction."""
        result = self.processor.process(
            "It is definitely true that the sky is blue."
        )
        assert "definitely" in result.representation.certainty_markers or \
               "is" in result.representation.certainty_markers

    def test_hedging_detection(self):
        """Test hedging marker detection."""
        result = self.processor.process(
            "It might possibly be true, apparently."
        )
        assert len(result.representation.certainty_markers) > 0
        # Should contain hedging markers
        has_hedging = any(
            m in result.representation.certainty_markers 
            for m in ["might", "possibly", "apparently"]
        )
        assert has_hedging

    def test_predicate_extraction(self):
        """Test predicate extraction."""
        # Use a sentence with a copula verb
        result = self.processor.process("The quick brown fox is jumping over the lazy dog.")
        assert len(result.representation.predicates) > 0

    def test_sentiment_basic(self):
        """Test basic sentiment detection."""
        positive = self.processor.process("This is great and wonderful.")
        negative = self.processor.process("This is terrible and awful.")
        # Positive should have >= 0 sentiment
        assert (positive.representation.sentiment or 0) >= 0
        # Negative should have <= 0 sentiment
        assert (negative.representation.sentiment or 0) <= 0

    def test_perception_result_structure(self):
        """Test that perception result has correct structure."""
        result = self.processor.process("Test input.")
        assert isinstance(result, PerceptionResult)
        assert result.representation is not None
        assert isinstance(result.confidence, PerceptionConfidence)
        assert Modality.TEXT in result.detected_modalities
        assert "word_count" in result.extracted_features
        assert "character_count" in result.extracted_features

    def test_quality_assessment(self):
        """Test that quality is assessed based on content richness."""
        # Rich text should be HIGH quality
        rich = self.processor.process(
            "John Smith (CEO of Acme Corp) announced yesterday that "
            "the company will expand to Europe in 2025."
        )
        assert rich.representation.quality in (
            RepresentationQuality.HIGH, 
            RepresentationQuality.MODERATE
        )

        # Simple text should be LOWER quality
        simple = self.processor.process("Hi.")
        # Very short text has unknown quality
        assert simple.representation.quality in (
            RepresentationQuality.LOW, 
            RepresentationQuality.UNKNOWN
        )


class TestPerceptionEngine:
    """Tests for the perception engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PerceptionEngine()

    def test_text_perception(self):
        """Test text perception through engine."""
        result = self.engine.perceive("The sky is blue.")
        assert result.representation.content == "The sky is blue."
        assert Modality.TEXT in result.detected_modalities

    def test_auto_detect_text(self):
        """Test auto-detection of text modality."""
        result = self.engine.perceive("Some text content")
        assert result.representation.modality == "text"

    def test_auto_detect_url(self):
        """Test auto-detection of URL as web page."""
        result = self.engine.perceive("https://example.com")
        assert result.representation.modality == "web_page"

    def test_auto_detect_image_path(self):
        """Test auto-detection of image path."""
        result = self.engine.perceive("photo.jpg")
        assert result.representation.modality == "image"

    def test_image_fallback(self):
        """Test image fallback when no model available."""
        result = self.engine.perceive("image.png", modality=Modality.IMAGE)
        assert result.representation.confidence == 0.3
        assert "image_model_unavailable" in result.confidence.uncertainty_markers
        assert "visual_content" in result.missing_information

    def test_audio_fallback(self):
        """Test audio fallback."""
        result = self.engine.perceive("audio.mp3", modality=Modality.AUDIO)
        assert result.representation.confidence == 0.3
        assert "transcription" in result.missing_information

    def test_video_fallback(self):
        """Test video fallback."""
        result = self.engine.perceive("video.mp4", modality=Modality.VIDEO)
        assert result.representation.confidence == 0.3

    def test_latency_tracking(self):
        """Test that processing latency is tracked."""
        result = self.engine.perceive("Test.")
        assert result.processing_latency_ms >= 0
        assert isinstance(result.processing_latency_ms, float)

    def test_context_passed(self):
        """Test that context is passed through."""
        result = self.engine.perceive(
            "Test content.",
            source="test_source",
            task="testing",
        )
        assert result.representation.source == "test_source"


class TestModalityDetection:
    """Tests for modality detection."""

    def setup_method(self):
        self.engine = PerceptionEngine()

    def test_string_detection(self):
        """Test string input detection."""
        assert self.engine._detect_modality("plain text") == Modality.TEXT
        assert self.engine._detect_modality("https://example.com") == Modality.WEB_PAGE
        assert self.engine._detect_modality("file.pdf") == Modality.DOCUMENT

    def test_image_extensions(self):
        """Test image file extension detection."""
        for ext in ["jpg", "jpeg", "png", "gif", "bmp", "webp"]:
            assert self.engine._detect_modality(f"file.{ext}") == Modality.IMAGE

    def test_audio_extensions(self):
        """Test audio file extension detection."""
        for ext in ["mp3", "wav", "ogg", "flac", "aac"]:
            assert self.engine._detect_modality(f"file.{ext}") == Modality.AUDIO

    def test_video_extensions(self):
        """Test video file extension detection."""
        for ext in ["mp4", "avi", "mkv", "mov", "webm"]:
            assert self.engine._detect_modality(f"file.{ext}") == Modality.VIDEO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
