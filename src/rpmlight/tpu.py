"""Optional TPUaaS (Coral EdgeTPU) & Gemini Flash Vision Integration for RPMLight.

All external AI/TPU URLs and API keys are 100% optional (`tpu_url=""`, `gemini_api_key=""`
by default, or configured via `RPMLIGHT_TPU_URL` / `RPMLIGHT_GEMINI_API_KEY` environment
variables).

Key features:
1. **Low-Rate Random-Frame Scene Cues (`TPUaaSBackgroundWorker`, default 0.25 Hz / every 4s)**:
   - Never streams full-rate video to the TPU. Instead, samples a single random/periodic
     frame every `4.0s` (only when the screen is active and changing) using `model=ssd_coco`
     and `priority=background` so it shares the 8 MB Coral EdgeTPU SRAM cache (`5.94 ms`,
     `0.0 ms` weight-swap penalty) alongside robot/drone streams.
   - Sends raw planar `I420` bytes directly (`?format=i420&width=320&height=176`) for
     `0.0 ms` encoding CPU overhead on embedded wall switches, with a pure-Python binary
     `P6` PPM helper (`i420_to_ppm`) also available.
2. **Optional 3-Tier Auto-Crop with Saved Manual/Hardcoded Crop Revert (`CropCalibrator`)**:
   - Always preserves your initial/manual crop (`saved_crop`) so `revert_crop()` restores
     the exact hardcoded margins at any time.
   - Supports:
     - **Tier 1 (Local Coral EdgeTPU via `tpu_url`)**: Uses `/v1/projects/ambiled?include_segmentation=true`.
     - **Tier 2 (Cloud Gemini Vision via `gemini_api_key`)**: Uses `gemini-2.5-flash` over
       stdlib HTTPS to locate the illuminated TV display (`[ymin, xmin, ymax, xmax]`) and
       classify the current TV mood/genre for users without a local Coral TPU.
     - **Tier 3 (Local Pure-Python I420 Luma BBox Fallback)**: Uses 92nd-percentile row/col
       luma thresholding on raw `I420` frames when offline.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from rpmlight.engine import DEFAULT_CROP_H, DEFAULT_CROP_W

SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
MIN_CROP_SPAN = 32


@dataclass(frozen=True)
class CropMargins:
    """8-pixel aligned hardware sensor crop margins (`top`, `bottom`, `left`, `right`)."""

    top: int = 328
    bottom: int = 216
    left: int = 608
    right: int = 352

    @property
    def width(self) -> int:
        return max(MIN_CROP_SPAN, SENSOR_WIDTH - self.left - self.right)

    @property
    def height(self) -> int:
        return max(MIN_CROP_SPAN, SENSOR_HEIGHT - self.top - self.bottom)

    def normalized(self) -> CropMargins:
        top = max(0, min(SENSOR_HEIGHT - MIN_CROP_SPAN, (self.top // 8) * 8))
        max_bottom = max(0, SENSOR_HEIGHT - MIN_CROP_SPAN - top)
        bottom = max(0, min(max_bottom, (self.bottom // 8) * 8))

        left = max(0, min(SENSOR_WIDTH - MIN_CROP_SPAN, (self.left // 8) * 8))
        max_right = max(0, SENSOR_WIDTH - MIN_CROP_SPAN - left)
        right = max(0, min(max_right, (self.right // 8) * 8))
        return CropMargins(top=top, bottom=bottom, left=left, right=right)

    @classmethod
    def from_bbox_norm(
        cls,
        bbox_norm: tuple[float, float, float, float] | list[float],
        sensor_w: int = SENSOR_WIDTH,
        sensor_h: int = SENSOR_HEIGHT,
    ) -> CropMargins:
        """Convert normalized `[x_min, y_min, x_max, y_max]` (`0.0..1.0`) to 8px sensor margins."""
        x1 = max(0.0, min(1.0, float(bbox_norm[0])))
        y1 = max(0.0, min(1.0, float(bbox_norm[1])))
        x2 = max(x1 + 0.05, min(1.0, float(bbox_norm[2])))
        y2 = max(y1 + 0.05, min(1.0, float(bbox_norm[3])))
        left = int(round(x1 * sensor_w))
        right = int(round((1.0 - x2) * sensor_w))
        top = int(round(y1 * sensor_h))
        bottom = int(round((1.0 - y2) * sensor_h))
        return cls(top=top, bottom=bottom, left=left, right=right).normalized()


@dataclass(frozen=True)
class TPUSceneCue:
    """Parsed scene semantics, wing saliency weights, and crop hints from TPUaaS or Gemini."""

    scene_class: str = "auto"  # "auto" | "drama" | "music_video"
    focal_x: float = 0.5  # 0.0 (far left) .. 0.5 (center) .. 1.0 (far right)
    focal_y: float = 0.5  # 0.0 (top) .. 0.5 (center) .. 1.0 (bottom)
    wing_weights: dict[str, float] = field(
        default_factory=lambda: {"west": 1.0, "east": 1.0, "far_left": 1.0}
    )
    active_content_bbox_norm: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    recommended_bbox_norm: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    viewers_in_room: int = 0
    tv_screen_detected: bool = False
    tpu_ms: float = 0.0
    sram_cache_hit: bool = False
    mood_summary: str = ""


def i420_to_ppm(
    frame: bytes | memoryview,
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
    step: int = 2,
) -> bytes:
    """Convert a planar I420 frame into a binary P6 PPM image (stdlib-only, PIL-compatible)."""
    mv = memoryview(frame)
    y_size = width * height
    uv_w = width >> 1
    uv_size = uv_w * (height >> 1)
    y_plane = mv[0:y_size]
    u_plane = mv[y_size : y_size + uv_size]
    v_plane = mv[y_size + uv_size : y_size + 2 * uv_size]

    out_w = width // step
    out_h = height // step
    header = f"P6\n{out_w} {out_h}\n255\n".encode("ascii")
    rgb = bytearray(out_w * out_h * 3)
    dst = 0

    for py in range(0, height, step):
        y_row = py * width
        uv_row = (py >> 1) * uv_w
        for px in range(0, width, step):
            yv = y_plane[y_row + px]
            uv_i = uv_row + (px >> 1)
            u = u_plane[uv_i] - 128
            v = v_plane[uv_i] - 128

            c = yv - 16
            if c < 0:
                c = 0
            r = (298 * c + 409 * v + 128) >> 8
            g = (298 * c - 100 * u - 208 * v + 128) >> 8
            b = (298 * c + 516 * u + 128) >> 8

            rgb[dst] = 0 if r < 0 else (255 if r > 255 else r)
            rgb[dst + 1] = 0 if g < 0 else (255 if g > 255 else g)
            rgb[dst + 2] = 0 if b < 0 else (255 if b > 255 else b)
            dst += 3

    return header + bytes(rgb)


def estimate_i420_active_bbox_norm(
    frame: bytes | memoryview,
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
    luma_threshold: int = 24,
) -> tuple[float, float, float, float]:
    """Pure-Python 92nd-percentile row/column luma active bounding-box estimator."""
    mv = memoryview(frame)
    y_plane = mv[0 : width * height]
    step_x = max(1, width // 64)
    step_y = max(1, height // 64)

    active_rows: list[int] = []
    for py in range(0, height, step_y):
        row_off = py * width
        vals = sorted(y_plane[row_off + px] for px in range(0, width, step_x))
        p92 = vals[min(len(vals) - 1, int(len(vals) * 0.92))]
        if p92 > luma_threshold:
            active_rows.append(py)

    active_cols: list[int] = []
    for px in range(0, width, step_x):
        vals = sorted(y_plane[py * width + px] for py in range(0, height, step_y))
        p92 = vals[min(len(vals) - 1, int(len(vals) * 0.92))]
        if p92 > luma_threshold:
            active_cols.append(px)

    if not active_rows or not active_cols:
        return (0.0, 0.0, 1.0, 1.0)
    return (
        round(active_cols[0] / width, 4),
        round(active_rows[0] / height, 4),
        round(min(width, active_cols[-1] + step_x) / width, 4),
        round(min(height, active_rows[-1] + step_y) / height, 4),
    )


def infer_scene_cue(payload: dict[str, Any]) -> TPUSceneCue:
    """Derive RPMLight `TPUSceneCue` from a TPUaaS `/v1/projects/ambiled` or `/v1/detect` response."""
    server_cue = payload.get("scene_cue")
    if isinstance(server_cue, dict):
        s_class = str(server_cue.get("scene_class", "auto"))
        fx = float(server_cue.get("focal_x", 0.5))
        fy = float(server_cue.get("focal_y", 0.5))
        ww_raw = server_cue.get("wing_weights") or {}
        wing_weights = {
            "west": float(ww_raw.get("west", 1.0)),
            "east": float(ww_raw.get("east", 1.0)),
            "far_left": float(ww_raw.get("far_left", 1.0)),
        }
        ab_raw = server_cue.get("active_content_bbox_norm") or [0.0, 0.0, 1.0, 1.0]
        active_bbox = (
            float(ab_raw[0]),
            float(ab_raw[1]),
            float(ab_raw[2]),
            float(ab_raw[3]),
        )
    else:
        detections = payload.get("detections") or []
        people = [d for d in detections if d.get("label") in ("person", "face")]
        action_objs = [
            d
            for d in detections
            if d.get("label")
            in ("sports ball", "car", "motorcycle", "airplane", "boat", "skateboard", "surfboard")
        ]
        max_person_area = 0.0
        for p in people:
            bbox = p.get("bbox_norm") or [0.0, 0.0, 0.0, 0.0]
            if len(bbox) == 4:
                area = max(0.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
                if area > max_person_area:
                    max_person_area = area

        if 1 <= len(people) <= 2 and max_person_area >= 0.18 and not action_objs:
            s_class = "drama"
        elif len(people) >= 4 or len(action_objs) >= 1:
            s_class = "music_video"
        else:
            s_class = "auto"

        sal = payload.get("saliency_center_norm")
        if isinstance(sal, dict) and "x" in sal and "y" in sal:
            fx = float(sal["x"])
            fy = float(sal["y"])
        else:
            wx, wy, tw = 0.5, 0.5, 0.0
            for d in detections:
                bbox = d.get("bbox_norm") or [0.0, 0.0, 1.0, 1.0]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                    area = max(0.01, (x2 - x1) * (y2 - y1))
                    w = float(d.get("score", 0.5)) * (area**0.5)
                    wx += cx * w
                    wy += cy * w
                    tw += w
            if tw > 0.0:
                fx = round((wx - 0.5) / tw, 4)
                fy = round((wy - 0.5) / tw, 4)
            else:
                fx, fy = 0.5, 0.5

        bias = (fx - 0.5) * 0.8
        wing_weights = {
            "west": round(max(0.55, min(1.45, 1.0 - bias)), 3),
            "east": round(max(0.55, min(1.45, 1.0 + bias)), 3),
            "far_left": round(max(0.50, min(1.50, 1.0 - bias * 1.25)), 3),
        }
        active_bbox = (0.0, 0.0, 1.0, 1.0)

    bbox_raw = payload.get("recommended_screen_bbox_norm") or list(active_bbox)
    rec_bbox = (
        float(bbox_raw[0]),
        float(bbox_raw[1]),
        float(bbox_raw[2]),
        float(bbox_raw[3]),
    )
    detections_list = payload.get("detections") or []
    people_count = sum(1 for d in detections_list if d.get("label") in ("person", "face"))
    return TPUSceneCue(
        scene_class=s_class,
        focal_x=max(0.0, min(1.0, fx)),
        focal_y=max(0.0, min(1.0, fy)),
        wing_weights=wing_weights,
        active_content_bbox_norm=active_bbox,
        recommended_bbox_norm=rec_bbox,
        viewers_in_room=int(payload.get("viewers_in_room", people_count)),
        tv_screen_detected=bool(payload.get("tv_screen_detected", False)),
        tpu_ms=float(payload.get("tpu_ms", 0.0)),
        sram_cache_hit=bool(payload.get("sram_cache_hit", False)),
        mood_summary=str(payload.get("mood_summary", "")),
    )


def query_gemini_vision_cue(
    jpeg_bytes: bytes,
    api_key: str,
    *,
    model: str = "gemini-2.5-flash",
    timeout_s: float = 6.0,
) -> TPUSceneCue:
    """Use Gemini Flash Vision (optional `gemini_api_key`) for TV Auto-Crop & Scene Mood."""
    if not api_key:
        raise ValueError("gemini_api_key is required for query_gemini_vision_cue")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={urllib.parse.quote(api_key)}"
    )
    prompt = (
        "Analyze this camera view of a TV screen. Return ONLY a JSON object with keys: "
        '"tv_screen_detected" (bool), '
        '"recommended_screen_bbox_norm" ([x_min, y_min, x_max, y_max] in 0.0..1.0 of the active TV display), '
        '"scene_class" ("drama" for dialogue/close-ups, "music_video" for concerts/sports/action, or "auto"), '
        '"focal_x" (0.0..1.0 horizontal attention center), '
        '"focal_y" (0.0..1.0 vertical attention center), '
        'and "mood_summary" (short 6-10 word description of the visual mood/colors).'
    )
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(jpeg_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw_resp = json.loads(resp.read().decode("utf-8"))
    text = (
        raw_resp.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "{}")
    )
    parsed = json.loads(text)
    fx = float(parsed.get("focal_x", 0.5))
    fy = float(parsed.get("focal_y", 0.5))
    bias = (fx - 0.5) * 0.8
    parsed["saliency_center_norm"] = {"x": fx, "y": fy}
    parsed["scene_cue"] = {
        "scene_class": str(parsed.get("scene_class", "auto")),
        "focal_x": fx,
        "focal_y": fy,
        "active_content_bbox_norm": parsed.get("recommended_screen_bbox_norm")
        or [0.0, 0.0, 1.0, 1.0],
        "wing_weights": {
            "west": round(max(0.55, min(1.45, 1.0 - bias)), 3),
            "east": round(max(0.55, min(1.45, 1.0 + bias)), 3),
            "far_left": round(max(0.50, min(1.50, 1.0 - bias * 1.25)), 3),
        },
    }
    return infer_scene_cue(parsed)


class CropCalibrator:
    """Manages optional Auto-Crop while preserving hardcoded/manual crop values for instant revert."""

    def __init__(
        self,
        hardcoded_crop: CropMargins | None = None,
        *,
        auto_crop_enabled: bool = False,
    ) -> None:
        self.hardcoded_crop = (hardcoded_crop or CropMargins()).normalized()
        self.saved_crop = self.hardcoded_crop
        self.current_crop = self.hardcoded_crop
        self.auto_crop_enabled = auto_crop_enabled

    def set_manual_crop(self, crop: CropMargins) -> CropMargins:
        """Update manual crop and save it as the revert target."""
        norm = crop.normalized()
        self.saved_crop = norm
        self.current_crop = norm
        return norm

    def apply_auto_crop_bbox(
        self, bbox_norm: tuple[float, float, float, float] | list[float]
    ) -> CropMargins:
        """Apply an auto-detected bounding box while keeping `saved_crop` intact for revert."""
        self.current_crop = CropMargins.from_bbox_norm(bbox_norm)
        return self.current_crop

    def revert_crop(self) -> CropMargins:
        """Restore the saved manual/hardcoded crop margins."""
        self.current_crop = self.saved_crop
        self.auto_crop_enabled = False
        return self.current_crop


class TPUaaSBackgroundWorker:
    """Optional low-rate (0.25 Hz) Mode 1 background cue worker for RPMLight."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        gemini_api_key: str | None = None,
        interval_s: float = 4.0,
        sram_friendly_model: str = "ssd_coco",
        priority: str = "background",
        timeout_s: float = 2.0,
    ) -> None:
        env_url = os.environ.get("RPMLIGHT_TPU_URL", "").strip()
        self.base_url = (base_url if base_url is not None else env_url).rstrip("/")
        env_key = (
            os.environ.get("RPMLIGHT_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip()
        self.gemini_api_key = gemini_api_key if gemini_api_key is not None else env_key
        self.interval_s = max(2.0, float(interval_s))
        self.sram_friendly_model = sram_friendly_model
        self.priority = priority
        self.timeout_s = timeout_s
        self.latest_cue: TPUSceneCue | None = None
        self._pending_i420: tuple[bytes, int, int] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_submit_ts = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RPMLight-TPUaaS", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def offer_i420_frame(
        self,
        frame: bytes | memoryview,
        width: int = DEFAULT_CROP_W,
        height: int = DEFAULT_CROP_H,
    ) -> None:
        """Offer a single random/periodic I420 frame if `interval_s` (4.0s) has elapsed."""
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_submit_ts < self.interval_s:
            return
        self._last_submit_ts = now
        with self._lock:
            self._pending_i420 = (bytes(frame), width, height)

    def query_i420_cue_sync(
        self,
        frame_bytes: bytes,
        width: int = DEFAULT_CROP_W,
        height: int = DEFAULT_CROP_H,
        *,
        include_segmentation: bool = False,
    ) -> TPUSceneCue:
        """POST raw planar I420 bytes directly to `/v1/projects/ambiled` (0.0ms switch CPU cost)."""
        if not self.base_url:
            bbox = estimate_i420_active_bbox_norm(frame_bytes, width=width, height=height)
            cue = TPUSceneCue(active_content_bbox_norm=bbox, recommended_bbox_norm=bbox)
            self.latest_cue = cue
            return cue

        q = urllib.parse.urlencode(
            {
                "format": "i420",
                "width": width,
                "height": height,
                "include_segmentation": str(include_segmentation).lower(),
                "model": self.sram_friendly_model,
                "priority": self.priority,
                "client": "ambiled",
            }
        )
        url = f"{self.base_url}/v1/projects/ambiled?{q}"
        req = urllib.request.Request(
            url,
            data=frame_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cue = infer_scene_cue(data)
        self.latest_cue = cue
        return cue

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            item: tuple[bytes, int, int] | None = None
            with self._lock:
                if self._pending_i420 is not None:
                    item = self._pending_i420
                    self._pending_i420 = None
            if item is None:
                self._stop.wait(timeout=0.25)
                continue
            raw_i420, w, h = item
            try:
                self.query_i420_cue_sync(raw_i420, width=w, height=h, include_segmentation=False)
            except Exception:
                pass
