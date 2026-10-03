"""Vision integrations: face detection and recognition providers.

Multiple face-analysis backends are supported; they are probed lazily
and reported honestly when the local machine cannot host them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sweep_core.integrations import _module_available


def availability() -> dict[str, Any]:
    return {
        "opencv": {"available": _module_available("cv2")},
        "yunet": {"available": _module_available("cv2") and hasattr(__import__("cv2"), "FaceDetectorYN")},
        "deepface": _probe_with_reason("deepface"),
        "torch": {"available": _module_available("torch")},
        "onnxruntime": {"available": _module_available("onnxruntime")},
        "insightface": _probe_with_reason("insightface"),
        "face_recognition": _probe_with_reason("face_recognition"),
        "faceaisdk_android": {
            "available": False,
            "reason": "Android-only SDK; not applicable to the Python service",
        },
    }


def detect_faces(image_path: str) -> dict[str, Any]:
    """Detect frontal faces using OpenCV 5.0+ YuNet or legacy Haar cascade.

    The YuNet ONNX model (~220KB) is downloaded once on first use into
    models/face_yunet.onnx and never needs re-fetching.
    """
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"unreadable image: {image_path}")

    yunet_path = Path("models/face_yunet.onnx")
    if hasattr(cv2, "FaceDetectorYN"):
        if not yunet_path.exists():
            yunet_path.parent.mkdir(parents=True, exist_ok=True)
            _download(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                yunet_path,
            )
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 320), 0.6
        )
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        status, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            return {
                "provider": "opencv-yunet",
                "faces": len(faces),
                "boxes": [[int(v) for v in f[:4]] for f in faces],
            }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return {
        "provider": "opencv-haar",
        "faces": len(boxes),
        "boxes": [[int(v) for v in box] for box in boxes],
    }


def _download(url: str, dest: Path) -> None:
    import ssl
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "SweepVision/0.1"})
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        dest.write_bytes(resp.read())


def _probe_with_reason(module: str) -> dict[str, Any]:
    if not _module_available(module):
        return {
            "available": False,
            "reason": f"{module} not installed (native build may be required)",
        }
    try:
        __import__(module)
        return {"available": True}
    except Exception as exc:
        return {"available": False, "reason": f"import failed: {exc}"}


def analyze_faces(image_path: str, provider: str = "auto") -> dict[str, Any]:
    """Detect faces in an image with the first available provider."""
    if provider in ("auto", "opencv"):
        try:
            return detect_faces(image_path)
        except Exception:
            pass
    if provider in ("auto", "insightface") and _module_available("insightface"):
        return _analyze_with_face_encoder(image_path)
    if provider in ("auto", "face_recognition") and _module_available("face_recognition"):
        import face_recognition

        locations = face_recognition.face_locations(face_recognition.load_image_file(image_path))
        return {"provider": "face_recognition", "faces": len(locations), "boxes": locations}
    raise RuntimeError("no face-analysis provider available in this environment")


def _analyze_with_face_encoder(image_path: str) -> dict[str, Any]:
    """Face analysis using an ONNX-based face encoder backend."""
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    frame = cv2.imread(image_path)
    faces = app.get(frame)
    return {"provider": "insightface", "faces": len(faces)}
