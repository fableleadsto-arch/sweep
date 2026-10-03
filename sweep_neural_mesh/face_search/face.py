"""Face pipeline: detect → crop → embed (512-d ArcFace).

Backends, tried in order:
  1. insightface buffalo_l (det_10g + w600k_r50 ONNX) — detection AND 512-d
     embeddings. Weights ship in models/face/insightface (already downloaded
     by services/intelligence/model_downloader.py); the pack is registered
     with providers=["CPUExecutionProvider"] so no GPU is needed.
  2. OpenCV YuNet (FaceDetectorYN) — detection only; embeddings come from the
     ONNX w600k_r50 recognizer directly (no insightface import needed).
  3. Haar cascade — last-resort detection only, no embeddings.

Everything is lazy: nothing heavy is imported until a search actually runs,
and every failure degrades to a lower-capability mode instead of raising.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from .models import Face

logger = logging.getLogger("sweep.face_search")

_REPO_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL_DIR = Path(
    __import__("os").environ.get("SWEEP_FACE_MODELS", _REPO_ROOT / "models" / "face" / "insightface")
)
# ArcFace w600k_r50 single ONNX (used without insightface as backend 2)
_ARCFACE_ONNX_CANDIDATES = ("w600k_r50.onnx", "glintr100.onnx")

_INSIGHTFACE_KW = dict(det_size=(640, 640), providers=["CPUExecutionProvider"])

_lock = threading.Lock()
_app: Any = None  # insightface FaceAnalysis, built once
_arc_session: Any = None  # onnxruntime InferenceSession for the recognizer


def availability() -> dict[str, Any]:
    """Honest capability report for this host (cheap, no model loads)."""
    return {
        "insightface": {
            "installed": _spec("insightface"),
            "weights": (FACE_MODEL_DIR / "models" / "buffalo_l").is_dir(),
        },
        "onnxruntime": {"installed": _spec("onnxruntime")},
        "opencv": {"installed": _spec("cv2")},
    }


def detect(image_path: str) -> list[Face]:
    """Detect faces in an image. Returns [] when no face is present."""
    img = _imread(image_path)
    if img is None:
        raise FileNotFoundError(f"unreadable image: {image_path}")
    errors: list[str] = []
    try:
        return _detect_insightface(img)
    except Exception as exc:  # noqa: BLE001 — degrade to next backend
        errors.append(f"insightface: {exc}")
    try:
        return _detect_yunet(img)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"yunet: {exc}")
    try:
        return _detect_haar(img)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"haar: {exc}")
    raise RuntimeError("no face-detection backend available: " + "; ".join(errors))


def largest_face(image_path: str) -> Optional[Face]:
    """The dominant (largest-area) face, embedded when a backend allows."""
    faces = detect(image_path)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.area)
    return embed_face(image_path, face)


def embed_face(image_path: str, face: Face) -> Face:
    """Attach a 512-d embedding to a detected face (best effort)."""
    img = _imread(image_path)
    if img is None:
        return face
    x1, y1, x2, y2 = face.box
    crop = _crop_aligned(img, face)
    if crop is None:
        return face
    # 1. insightface handles crop-alignment + recognition itself
    if _spec("insightface"):
        try:
            app = _get_insightface()
            for f in app.get(img):
                bx = tuple(int(v) for v in f.bbox[:4])
                if _iou(bx, face.box) > 0.5 and getattr(f, "normed_embedding", None) is not None:
                    face.embedding = [float(v) for v in f.normed_embedding]
                    return face
        except Exception as exc:  # noqa: BLE001
            logger.debug("insightface embed failed: %s", exc)
    # 2. raw ONNX recognizer on our own crop
    try:
        emb = _embed_onnx(crop)
        if emb is not None:
            face.embedding = emb
    except Exception as exc:  # noqa: BLE001
        logger.debug("onnx embed failed: %s", exc)
    return face


def cosine(a: Optional[list[float]], b: Optional[list[float]]) -> Optional[float]:
    """Cosine similarity between two embeddings (None when unusable)."""
    if not a or not b or len(a) != len(b):
        return None
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


# ── detection backends ────────────────────────────────────────────────


def _detect_insightface(img: Any) -> list[Face]:
    app = _get_insightface()
    faces = []
    for f in app.get(img):
        box = tuple(int(v) for v in f.bbox[:4])
        faces.append(Face(box=box, det_score=float(f.det_score)))
    return faces


def _detect_yunet(img: Any) -> list[Face]:
    import cv2

    model_path = Path("models/face_yunet.onnx")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _download(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx",
            model_path,
        )
    h, w = img.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.6)
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    out = []
    for f in faces or []:
        out.append(Face(box=(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])), det_score=float(f[14])))
    return out


def _detect_haar(img: Any) -> list[Face]:
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [
        Face(box=(int(x), int(y), int(x + w), int(y + h)), det_score=0.6)
        for (x, y, w, h) in boxes
    ]


# ── insightface singleton ─────────────────────────────────────────────


def _get_insightface() -> Any:
    global _app
    if _app is not None:
        return _app
    with _lock:
        if _app is None:
            from insightface.app import FaceAnalysis

            # FaceAnalysis resolves <root>/models/<name> at construction time,
            # so pass the repo's pack location when it exists — nothing is
            # re-downloaded and the engine is self-contained.
            if (FACE_MODEL_DIR / "models" / "buffalo_l").is_dir():
                app = FaceAnalysis(
                    name="buffalo_l",
                    root=str(FACE_MODEL_DIR),
                    providers=["CPUExecutionProvider"],
                )
            else:
                app = FaceAnalysis(
                    name="buffalo_l", providers=["CPUExecutionProvider"]
                )
            app.prepare(ctx_id=-1, **_INSIGHTFACE_KW)
            _app = app
    return _app


# ── raw-ONNX ArcFace embedding (no insightface needed) ────────────────


def _embed_onnx(crop_bgr: Any) -> Optional[list[float]]:
    global _arc_session
    import cv2
    import numpy as np

    onnx_path = None
    pack_dir = FACE_MODEL_DIR / "models" / "buffalo_l"
    for name in _ARCFACE_ONNX_CANDIDATES:
        p = pack_dir / name
        if p.exists():
            onnx_path = p
            break
    if onnx_path is None:
        return None
    if _arc_session is None:
        import onnxruntime as ort

        _arc_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    resized = cv2.resize(crop_bgr, (112, 112))
    blob = cv2.dnn.blobFromImage(resized, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    input_name = _arc_session.get_inputs()[0].name
    emb = _arc_session.run(None, {input_name: blob})[0][0]
    emb = np.asarray(emb, dtype=np.float64)
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        return None
    return [float(v) / norm for v in emb]


# ── small utilities ───────────────────────────────────────────────────


def _crop_aligned(img: Any, face: Face) -> Optional[Any]:
    """Crop with a 20% margin, clamped to image bounds."""
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.box
    mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _imread(path: str) -> Optional[Any]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def _download(url: str, dest: Path) -> None:
    import ssl
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "SweepFaceSearch/1.0"})
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        dest.write_bytes(resp.read())


def _spec(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
