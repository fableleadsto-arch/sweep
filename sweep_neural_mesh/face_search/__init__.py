"""Sweep Face/Reverse-Image Search.

Implementation of the openface-search + selfwatch pattern, extended to all
sources (not just social media) and integrated into the Sweep platform.

Quick start::

    from sweep_neural_mesh.face_search import FaceSearchOrchestrator

    engine = FaceSearchOrchestrator()
    report = engine.run_search("photo.jpg", name_hint="jane dough")
    print(report.to_dict())

Environment variables:
  SERPAPI_KEY              — enables Google Lens, Yandex, Bing, Google Images
  TINEYE_API_KEY           — TinEye public key
  TINEYE_PRIVATE_KEY       — TinEye HMAC private key
  SWEEP_PUBLIC_BASE_URL    — public https base of this host; enables
                             URL-based providers by serving face crops at
                             <base>/uploads/face_search/<file>.jpg
  SWEEP_UPLOADS_DIR        — where crops are written for static serving
                             (default: <tmp>/sweep_face_search_uploads)

Privacy: query images and crops are temp files, deleted after each run;
no face database is built; only public internet content is queried.
"""
from .models import (
    Face,
    ProfileMatch,
    ProviderResult,
    RawMatch,
    SearchReport,
)
from .face import availability as face_availability, cosine as face_cosine, detect, largest_face
from .orchestrator import FaceSearchOrchestrator, build_providers
from .social_parser import classify, PLATFORMS

__all__ = [
    "Face",
    "ProfileMatch",
    "ProviderResult",
    "RawMatch",
    "SearchReport",
    "FaceSearchOrchestrator",
    "build_providers",
    "face_availability",
    "face_cosine",
    "detect",
    "largest_face",
    "classify",
    "PLATFORMS",
]
