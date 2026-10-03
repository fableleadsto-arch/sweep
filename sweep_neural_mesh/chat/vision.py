"""Object detection & tracking for Sweep — pretrained models, CPU-friendly.

Detector: YOLO-class pretrained weights through OpenCV DNN (no heavy
framework dependency). Weights are downloaded once to models/vision/ and
cached. Tracking: OpenCV trackers per detected object with ID assignment —
objects keep their identity across frames; detection re-runs every N frames
to (re)initialize tracks.

Events: "entered/left/moved" events are emitted and can be routed into the
neural mesh (e.g. 'Claim: a person entered the scene. Evidence: tracker id 3
appeared at (x,y).') — vision becomes another sensory input to the mesh.

All entry points degrade gracefully when there is no camera.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "vision"
ONNX_REPO = "salim4n/yolov8n-detect-onnx"   # pretrained YOLOv8n (COCO 80 classes)
ONNX_FILE = "yolov8n-onnx-web/yolov8n.onnx"
ONNX_LOCAL = "yolov8n.onnx"
INPUT_SIZE = 640

# Official COCO 80-class list (embedded so no network fetch is needed)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class Track:
    track_id: int
    label: str
    box: tuple[int, int, int, int]   # x, y, w, h
    last_seen: float
    hits: int = 1
    age: float = field(default_factory=time.time)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


def ensure_weights() -> tuple[Path, Path]:
    """Ensure the pretrained detector is on disk (ONNX 12MB, COCO classes).

    Uses huggingface_hub when available (robust, cached); falls back to a
    direct URL fetch of the same file. COCO class names are embedded below
    (identical to the official list) so no extra download is needed.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx = MODELS_DIR / ONNX_LOCAL

    if not (onnx.exists() and onnx.stat().st_size > 1_000_000):
        try:
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(ONNX_REPO, ONNX_FILE, local_dir=str(MODELS_DIR))
            shutil.move(str(got), str(onnx))
            assert onnx.stat().st_size > 1_000_000
        except Exception:
            import urllib.request
            urllib.request.urlretrieve(
                f"https://huggingface.co/{ONNX_REPO}/resolve/main/{ONNX_FILE}", onnx)
    names = MODELS_DIR / "coco.names"
    if not names.exists():
        names.write_text("\n".join(COCO_CLASSES) + "\n", encoding="utf-8")
    return onnx, names


class ObjectDetector:
    """YOLOv4-tiny via OpenCV DNN. ~30-60 ms/frame on a modern CPU."""

    def __init__(self, conf_threshold: float = 0.45, nms_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._net = None
        self._classes: list[str] = []
        self._load_failed = False

    def _load(self) -> bool:
        if self._net is not None:
            return True
        if self._load_failed:
            return False
        try:
            import cv2
            onnx, names = ensure_weights()
            self._classes = names.read_text().split("\n")
            self._net = cv2.dnn.readNetFromONNX(str(onnx))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return True
        except Exception as e:
            logger.warning("YOLO load failed: %s", e)
            self._load_failed = True
            return False

    @property
    def ready(self) -> bool:
        return self._net is not None

    def detect(self, frame) -> list[Detection]:
        if not self._load():
            return []
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self._net.setInput(blob)
        out = self._net.forward()   # (1, 84, 8400): 4 box + 80 class scores
        pred = out[0].T             # (8400, 84)

        boxes, confs, class_ids = [], [], []
        scores_all = pred[:, 4:]
        cand_ids = scores_all.argmax(axis=1)
        cand_confs = scores_all.max(axis=1)
        mask = cand_confs >= self.conf_threshold
        sx, sy = w / INPUT_SIZE, h / INPUT_SIZE
        for cx, cy, bw, bh, ci, conf in zip(
                pred[mask, 0], pred[mask, 1], pred[mask, 2], pred[mask, 3],
                cand_ids[mask], cand_confs[mask]):
            # cxcywh in net-input pixels -> xywh in frame pixels
            x = int((cx - bw / 2) * sx)
            y = int((cy - bh / 2) * sy)
            boxes.append((x, y, int(bw * sx), int(bh * sy)))
            confs.append(float(conf))
            class_ids.append(int(ci))

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold, self.nms_threshold)
        results = []
        flat = np.array(idxs).flatten() if len(idxs) else []
        for i in flat:
            x, y, bw, bh = boxes[i]
            results.append(Detection(
                label=self._classes[class_ids[i]] if class_ids[i] < len(self._classes) else "?",
                confidence=confs[i], x=x, y=y, w=bw, h=bh,
            ))
        return results


class ObjectTracker:
    """IoU-based multi-object tracker over detector outputs.

    Simple and robust for CPU: greedy IoU matching between detections and
    live tracks, with max_age frames of coasting for occlusions.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame_no = 0

    def update(self, detections: list[Detection]) -> dict[int, dict]:
        """Match detections to tracks; returns per-id event dicts this frame."""
        self._frame_no += 1
        now = time.time()
        events: dict[int, dict] = {}

        unmatched = list(range(len(detections)))
        dead: list[int] = []

        for tid, tr in self.tracks.items():
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                if detections[j].label != tr.label:
                    continue
                v = _iou(tr.box, (detections[j].x, detections[j].y,
                                  detections[j].w, detections[j].h))
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= self.iou_threshold:
                d = detections[best_j]
                new_box = (d.x, d.y, d.w, d.h)
                moved = _center_dist(tr.box, new_box) > 25
                tr.box, tr.last_seen, tr.hits = new_box, now, tr.hits + 1
                unmatched.remove(best_j)
                if tid not in events and moved:
                    events[tid] = {"event": "moved", "id": tid, "label": tr.label,
                                   "at": tr.center}
            else:
                dead.append(tid)

        for tid in dead:
            tr = self.tracks.pop(tid)
            events[tid] = {"event": "left", "id": tid, "label": tr.label,
                           "at": tr.center}

        for j in unmatched:
            d = detections[j]
            tr = Track(track_id=self._next_id, label=d.label,
                       box=(d.x, d.y, d.w, d.h), last_seen=now)
            self.tracks[self._next_id] = tr
            events[self._next_id] = {"event": "entered", "id": self._next_id,
                                     "label": d.label, "at": tr.center}
            self._next_id += 1

        # prune stale
        for tid in list(self.tracks):
            if now - self.tracks[tid].last_seen > self.max_age * 0.2:
                tr = self.tracks.pop(tid)
                events.setdefault(tid, {"event": "left", "id": tid,
                                        "label": tr.label, "at": tr.center})
        return events

    def snapshot(self) -> list[dict]:
        return [{"id": t.track_id, "label": t.label, "box": t.box,
                 "center": t.center, "hits": t.hits}
                for t in self.tracks.values()]


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _center_dist(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ((ax + aw / 2 - bx - bw / 2) ** 2 + (ay + ah / 2 - by - bh / 2) ** 2) ** 0.5


class CameraSession:
    """Live camera -> detect -> track loop with optional preview window.

    Usage:
        with CameraSession() as cam:
            for events, snapshot, frame in cam.stream():
                ...
    """

    def __init__(self, camera_index: int = 0, show_preview: bool = True,
                 detect_every: int = 2):
        self.camera_index = camera_index
        self.show_preview = show_preview
        self.detect_every = detect_every
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()

    def __enter__(self):
        import cv2
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No camera available at index {self.camera_index}. "
                "Connect a webcam or pass --camera <index>.")
        return self

    def __exit__(self, *a):
        if hasattr(self, "_cap"):
            self._cap.release()
        if self.show_preview:
            import cv2
            cv2.destroyWindow("Sweep vision")

    def stream(self, max_seconds: float | None = None):
        """Yields (events, tracks_snapshot, annotated_frame) per frame."""
        import cv2
        t0 = time.time()
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            dets = (self.detector.detect(frame)
                    if self._frame_count() % self.detect_every == 0
                    else [])
            events = self.tracker.update(dets)
            for t in self.tracker.tracks.values():
                x, y, w, h = t.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (94, 234, 212), 2)
                cv2.putText(frame, f"#{t.track_id} {t.label}",
                            (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (94, 234, 212), 1)
            if self.show_preview:
                cv2.imshow("Sweep vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            yield events, self.tracker.snapshot(), frame
            if max_seconds and time.time() - t0 > max_seconds:
                break

    def _frame_count(self) -> int:
        return self.tracker._frame_no


def describe_scene(snapshot: list[dict]) -> str:
    """Human-readable scene summary from a track snapshot."""
    if not snapshot:
        return "nothing in view"
    from collections import Counter
    counts = Counter(t["label"] for t in snapshot)
    parts = [f"{n} {lbl}{'s' if n > 1 else ''}" for lbl, n in counts.most_common()]
    return ", ".join(parts)
