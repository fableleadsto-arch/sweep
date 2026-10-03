"""
Document Perception — Phase 2.

Handles perception of document-like inputs:
- Plain text files
- Structured documents (JSON, CSV, etc.)
- PDF-like text extraction (scaffolding - real PDF needs library)
- Markup documents (HTML, Markdown)

Current implementation is text-focused with structure detection.
Real PDF/image document processing requires external libraries.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from . import PerceptionResult

from ..representation import (
    Representation,
    RepresentationQuality,
    RepresentationType,
)
from sweep_cognitive.representation import (
    Representation,
    RepresentationQuality,
    RepresentationType,
)

# Import from parent module to avoid circular import
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sweep_cognitive.representation import (
    Representation,
    RepresentationQuality,
    RepresentationType,
)

# Import from parent module to avoid circular import
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Lazy import to avoid circular dependency
def _get_perception_classes():
    from sweep_cognitive.perception import Modality, PerceptionConfidence, PerceptionResult
    return Modality, PerceptionConfidence, PerceptionResult


@dataclass
class DocumentInfo:
    """Metadata about a perceived document."""
    title: str = ""
    author: str = ""
    created_date: str = ""
    modified_date: str = ""
    language: str = "unknown"
    word_count: int = 0
    character_count: int = 0
    page_count: int = 0  # Approximate
    structure: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "language": self.language,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "page_count": self.page_count,
            "structure": self.structure,
            "metadata": self.metadata,
        }


class DocumentPerceptionProcessor:
    """
    Processes document-like inputs into structured representations.
    
    Handles:
    - Plain text (with structure detection)
    - JSON (extracts structure and content)
    - CSV (detects tabular structure)
    - HTML/Markdown (detects markup structure)
    - Basic PDF-like handling (text extraction scaffolding)
    """

    # Structure patterns for different document types
    JSON_PATTERN = re.compile(r'^\s*[\{\[]\s*$', re.MULTILINE)
    CSV_PATTERN = re.compile(r'^[^\n]*,[^\n]*$')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    MARKDOWN_PATTERN = re.compile(r'^(#{1,6}|[-*+]|>|\|\s*\|)\s*')

    def process(self, input_data: Any, source: str = "unknown",
                **context):
        """
        Process document input and produce a structured representation.
        
        Args:
            input_data: The document content (string, path, or bytes)
            source: Where the document came from
            **context: Additional context
            
        Returns:
            PerceptionResult with document representation.
        """
        t0 = time.perf_counter()
        
        # Lazy import to avoid circular dependency
        Modality, PerceptionConfidence, PerceptionResult = _get_perception_classes()

        # Convert input to text
        text, doc_format = self._extract_text(input_data)
        
        if not text:
            rep = Representation(
                content="",
                representation_type=RepresentationType.PERCEPTUAL,
                quality=RepresentationQuality.NONE,
                confidence=0.0,
                source=source,
                modality="document",
            )
            return PerceptionResult(
                representation=rep,
                confidence=PerceptionConfidence(overall=0.0),
                detected_modalities=[Modality.DOCUMENT],
                missing_information=["content"],
            )

        # Detect document structure
        structure = self._detect_structure(text, doc_format)
        
        # Extract document info
        info = self._extract_document_info(text, structure, doc_format)
        
        # Determine quality
        quality = self._assess_quality(text, structure, info)
        
        # Build representation
        rep = Representation(
            content=text[:5000],  # Truncate very long documents
            representation_type=RepresentationType.PERCEPTUAL,
            quality=quality,
            confidence=self._confidence_from_quality(quality),
            source=source,
            modality="document",
            metadata={
                "document_format": doc_format,
                "document_info": info.to_dict(),
                "structure_types": [s.get("type", "unknown") for s in structure],
            },
        )

        # Missing information tracking
        missing = []
        if doc_format in ("pdf", "image"):
            missing.append(f"{doc_format}_extraction_incomplete")
        if not info.title and not info.metadata.get("title"):
            missing.append("title_not_detected")

        Modality, PerceptionConfidence, PerceptionResult = _get_perception_classes()
        latency = (time.perf_counter() - t0) * 1000
        
        return PerceptionResult(
            representation=rep,
            confidence=PerceptionConfidence(
                overall=self._confidence_from_quality(quality),
                modality_confidence={"document": self._confidence_from_quality(quality)},
                quality_markers=self._quality_markers(quality),
            ),
            detected_modalities=[Modality.DOCUMENT],
            extracted_features={
                "format": doc_format,
                "word_count": info.word_count,
                "structure_count": len(structure),
                "has_title": bool(info.title),
            },
            missing_information=missing,
            processing_latency_ms=latency,
        )

    def _extract_text(self, input_data: Any) -> tuple[str, str]:
        """
        Extract text from input data.
        
        Returns:
            Tuple of (text_content, detected_format)
        """
        if isinstance(input_data, str):
            # Could be raw text, JSON, CSV, HTML, path, etc.
            
            # Check if it's a file path
            if re.search(r'\.(txt|json|csv|html?|md|markdown|pdf)$', input_data, re.I):
                # It's a path - we can't actually read files here
                # Return scaffolding
                return f"[Document at path: {input_data}]", "path_reference"
            
            # Check format from content
            stripped = input_data.strip()
            
            # JSON detection
            if stripped.startswith('{') or stripped.startswith('['):
                try:
                    json.loads(stripped)
                    return stripped, "json"
                except json.JSONDecodeError:
                    pass
            
            # CSV detection (simple heuristic)
            lines = stripped.split('\n')
            if len(lines) > 1:
                csv_matches = sum(1 for line in lines if ',' in line and 
                                 len(line.split(',')) >= 2)
                if csv_matches > len(lines) * 0.5:
                    return stripped, "csv"
            
            # HTML detection
            if '<' in stripped and '>' in stripped and self.HTML_PATTERN.search(stripped):
                return stripped, "html"
            
            # Markdown detection
            if self.MARKDOWN_PATTERN.search(stripped):
                return stripped, "markdown"
            
            # Default: plain text
            return stripped, "text"
        
        elif isinstance(input_data, bytes):
            # Try to decode
            try:
                text = input_data.decode('utf-8')
                return text, "binary_text"
            except UnicodeDecodeError:
                try:
                    text = input_data.decode('latin-1')
                    return text, "binary_latin1"
                except:
                    return "", "binary_unreadable"
        
        elif isinstance(input_data, dict):
            # JSON-like dict
            return json.dumps(input_data, indent=2), "json_object"
        
        elif isinstance(input_data, list):
            # JSON-like list
            return json.dumps(input_data, indent=2), "json_array"
        
        else:
            return str(input_data), "unknown"

    def _detect_structure(self, text: str, doc_format: str) -> list[dict[str, Any]]:
        """
        Detect document structure (headings, sections, tables, etc.).
        
        This is scaffolding - real structure detection needs NLP.
        """
        structure = []
        
        if doc_format == "json":
            structure.append({
                "type": "json_object",
                "depth": self._json_depth(text),
                "note": "JSON structure detected",
            })
        elif doc_format == "csv":
            lines = text.split('\n')
            if lines:
                cols = len(lines[0].split(','))
                structure.append({
                    "type": "table",
                    "columns": cols,
                    "rows": len(lines),
                    "note": f"CSV table with ~{cols} columns",
                })
        elif doc_format in ("html",):
            # Count tags as structure heuristic
            headings = len(re.findall(r'<h[1-6][^>]*>', text, re.I))
            paragraphs = len(re.findall(r'<p[^>]*>', text, re.I))
            if headings:
                structure.append({
                    "type": "headings",
                    "count": headings,
                    "note": f"HTML document with {headings} headings",
                })
            if paragraphs:
                structure.append({
                    "type": "paragraphs",
                    "count": paragraphs,
                    "note": f"HTML document with {paragraphs} paragraphs",
                })
        elif doc_format in ("markdown",):
            headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
            lists = len(re.findall(r'^[-*+]\s', text, re.MULTILINE))
            if headings:
                structure.append({
                    "type": "headings",
                    "count": headings,
                    "note": f"Markdown with {headings} headings",
                })
        
        # Generic structure for any text
        if not structure:
            # Detect potential sections by paragraph count
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                structure.append({
                    "type": "paragraphs",
                    "count": len(paragraphs),
                    "note": f"Text with {len(paragraphs)} paragraphs",
                })
            
            # Detect potential list items
            list_items = len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE))
            if list_items > 0:
                structure.append({
                    "type": "list",
                    "count": list_items,
                    "note": f"Text with {list_items} list items",
                })
        
        return structure

    def _extract_document_info(self, text: str, structure: list,
                               doc_format: str) -> DocumentInfo:
        """Extract document metadata."""
        info = DocumentInfo()
        info.word_count = len(text.split())
        info.character_count = len(text)
        info.structure = structure
        
        info.metadata["format"] = doc_format
        info.metadata["extraction_method"] = "pattern_based"
        
        # Try to extract title (first line, often)
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and first_line:
                info.title = first_line
        
        # Count pages (crude: ~3000 chars per page)
        info.page_count = max(1, len(text) // 3000)
        
        return info

    def _json_depth(self, text: str) -> int:
        """Estimate JSON nesting depth."""
        max_depth = 0
        current_depth = 0
        for char in text:
            if char in '{[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in '}]':
                current_depth -= 1
        return max_depth

    def _assess_quality(self, text: str, structure: list,
                        info: DocumentInfo) -> RepresentationQuality:
        """Assess document perception quality."""
        score = 0
        
        if info.word_count > 0:
            score += 1
        if info.word_count > 50:
            score += 1
        if info.word_count > 500:
            score += 1
        if structure:
            score += 1
        if info.title:
            score += 1
        
        if score >= 4:
            return RepresentationQuality.HIGH
        elif score >= 2:
            return RepresentationQuality.MODERATE
        elif score >= 1:
            return RepresentationQuality.LOW
        return RepresentationQuality.UNKNOWN

    def _confidence_from_quality(self, quality: RepresentationQuality) -> float:
        """Map quality to confidence."""
        mapping = {
            RepresentationQuality.HIGH: 0.85,
            RepresentationQuality.MODERATE: 0.65,
            RepresentationQuality.LOW: 0.45,
            RepresentationQuality.UNKNOWN: 0.35,
            RepresentationQuality.NONE: 0.0,
        }
        return mapping.get(quality, 0.5)

    def _quality_markers(self, quality: RepresentationQuality) -> list[str]:
        """Return quality marker descriptions."""
        markers = {
            RepresentationQuality.HIGH: ["substantial_content", "structure_detected"],
            RepresentationQuality.MODERATE: ["adequate_content"],
            RepresentationQuality.LOW: ["limited_content"],
            RepresentationQuality.UNKNOWN: ["minimal_content"],
            RepresentationQuality.NONE: ["empty"],
        }
        return markers.get(quality, [])


class StructuredDataPerceptionProcessor:
    """
    Processes structured data inputs (JSON, CSV, tables).
    
    Extracts:
    - Schema information
    - Key-value relationships
    - Tabular structure
    - Data types (heuristic)
    """

    def process(self, input_data: Any, source: str = "unknown",
                **context) -> PerceptionResult:
        """
        Process structured data and produce a representation.
        
        Args:
            input_data: The structured data (dict, list, JSON string, etc.)
            source: Where the data came from
            **context: Additional context
            
        Returns:
            PerceptionResult with structured data representation.
        """
        t0 = time.perf_counter()
        
        Modality, PerceptionConfidence, PerceptionResult = _get_perception_classes()

        # Normalize to text representation
        text, data_type = self._normalize(input_data)
        
        if not text:
            rep = Representation(
                content="",
                representation_type=RepresentationType.PERCEPTUAL,
                quality=RepresentationQuality.NONE,
                confidence=0.0,
                source=source,
                modality="structured",
            )
            return PerceptionResult(
                representation=rep,
                confidence=PerceptionConfidence(overall=0.0),
                detected_modalities=[Modality.STRUCTURED],
            )
        
        # Extract structure information
        schema_info = self._extract_schema(input_data)
        
        # Build representation
        rep = Representation(
            content=text[:3000],
            representation_type=RepresentationType.PERCEPTUAL,
            quality=RepresentationQuality.MODERATE,  # Structured data is generally reliable
            confidence=0.75,
            source=source,
            modality="structured",
            metadata={
                "data_type": data_type,
                "schema": schema_info,
            },
        )
        
        latency = (time.perf_counter() - t0) * 1000
        
        Modality, PerceptionConfidence, PerceptionResult = _get_perception_classes()
        return PerceptionResult(
            representation=rep,
            confidence=PerceptionConfidence(
                overall=0.75,
                modality_confidence={"structured": 0.75},
            ),
            detected_modalities=[Modality.STRUCTURED],
            extracted_features={
                "data_type": data_type,
                "entry_count": schema_info.get("entry_count", 0),
                "key_count": len(schema_info.get("keys", [])),
            },
            processing_latency_ms=latency,
        )

    def _normalize(self, input_data: Any) -> tuple[str, str]:
        """Normalize input to text and detect type."""
        if isinstance(input_data, str):
            stripped = input_data.strip()
            if stripped.startswith('{') or stripped.startswith('['):
                return stripped, "json_string"
            if ',' in stripped and '\n' in stripped:
                return stripped, "csv_string"
            return stripped, "text_string"
        elif isinstance(input_data, dict):
            return json.dumps(input_data, indent=2), "dict"
        elif isinstance(input_data, list):
            return json.dumps(input_data, indent=2), "list"
        else:
            return str(input_data), "other"

    def _extract_schema(self, input_data: Any) -> dict[str, Any]:
        """Extract schema information from structured data."""
        schema = {
            "entry_count": 0,
            "keys": [],
            "types": {},
        }
        
        if isinstance(input_data, dict):
            schema["entry_count"] = 1
            schema["keys"] = list(input_data.keys())
            for key, value in input_data.items():
                schema["types"][key] = type(value).__name__
        elif isinstance(input_data, list):
            schema["entry_count"] = len(input_data)
            if input_data:
                if isinstance(input_data[0], dict):
                    schema["keys"] = list(input_data[0].keys())
                    for key in input_data[0]:
                        schema["types"][key] = type(input_data[0][key]).__name__
                else:
                    schema["types"]["items"] = type(input_data[0]).__name__
        
        return schema


__all__ = [
    "DocumentPerceptionProcessor",
    "StructuredDataPerceptionProcessor",
    "DocumentInfo",
]
