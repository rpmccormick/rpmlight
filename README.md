# RPMLight (`rpmlight`) — v0.4.0

**Perceptual Zero-Blink Multi-Zone Camera & Screen Ambilight Engine for Nanoleaf 4D, WLED, and Home Assistant Hue Sky / Wing Lights, with Optional Coral EdgeTPU (`TPUaaS`) & Gemini Vision Support.**

Traditional camera and screen-capture Ambilights (Hyperion, SignalRGB, Prismatik, stock Nanoleaf 4D) treat every zone as a raw RGB pixel mirror. In real living rooms, that causes **constant distracting brightness strobing**, **black-bar starvation on `2.35:1` widescreen movies & Kodi poster menus**, **blood-red skin/wood/shadow contamination**, and **flickering when actors walk across the frame**.

**RPMLight** solves these problems using a human-vision-inspired perceptual pipeline designed to run at **15–20 FPS in pure Python with <10% CPU usage** on embedded Linux / Android wall switches (such as the Brilliant Smart Home Control) or Raspberry Pi.

---

## Key Innovations (v0.4.0)

### 1. 3-Step Orthogonal Auto-Blackness Deletion (`2.35:1` Letterbox, Kodi Rows & `4:3` Pillarbox)
- Each of the 52 perimeter zones precomputes **3 orthogonal inward march steps**:
  - **`Step 0` (16:9 Outer Edge)**: Samples the true 16:9 perimeter (with bottom zones lifted to `y = 155..164` off the bottom TV bezel reflection).
  - **`Step 1` (Moderate Inward March)**: Steps `14px` deeper vertically on top/bottom (`y = 16..30` / `y = 140..152`) or `19px` deeper horizontally on left/right (`x = 22..38` / `x = 280..296`).
  - **`Step 2` (Deep Inward March)**: Steps `28px` deeper vertically (`y = 30..44` top / `y = 124..138` bottom) or `39px` deeper horizontally (`x = 42..58` left / `x = 260..276` right).
- Whenever an outer step reads black (`e_max < 32`), the zone automatically steps inward **orthogonally** (preserving exact horizontal `X` for top/bottom zones and exact vertical `Y` for left/right zones) so `2.35:1` widescreen movies, Kodi inset poster shelves, and `4:3` pillarboxed shows light up every zone at the default `50%` setting.

### 2. Optional Low-Rate Coral EdgeTPU (`TPUaaS`) & Gemini Flash Vision Cues
- **100% Optional (`tpu_url=""`, `gemini_api_key=""` by default)**:
  - Works completely standalone with zero external AI dependencies, or optionally connects to a local Coral EdgeTPU service (`RPMLIGHT_TPU_URL`) or Google Gemini Flash Vision (`RPMLIGHT_GEMINI_API_KEY`).
- **Low-Rate Random-Frame Scene Cues (`0.25 Hz` / every 4s, Zero Video Streaming)**:
  - Rather than streaming full video to the TPU, `TPUaaSBackgroundWorker` samples a single raw `I420` frame every `4.0 seconds` (`priority=background`, `model=ssd_coco`) so it uses `<1.5%` of a Coral EdgeTPU and shares the 8 MB SRAM cache (`5.94 ms` TPU time, `0 ms` weight swap) alongside companion robots and drones.
  - **Content-Aware Scene Mode (`scene_class`)**: Automatically stabilizes close-up dialogue (`"drama"`) against shot-reverse-shot brightness pumping, and unlocks `0 ms` beat/scene-cut snaps on concerts, sports, and action (`"music_video"`).
  - **Saliency-Weighted Overhead & Wing Balance (`wing_weights`)**: Gently biases West/East overhead sky lights and side wing lights toward the side of the screen where the main subject/action is located.
  - **Beat & Light-Show Choreography Telemetry (`ZoneFrameResult.to_diamond_dict()`)**: Publishes `beat_pulse`, `energy`, `is_scene_cut`, and `scene_class` alongside RGB channels so companion robots and smart lights can dance in sync with the light show.
- **Optional 3-Tier Auto-Crop with One-Click Revert (`CropCalibrator`)**:
  - Always preserves your hardcoded/manual sensor crop (`saved_crop`) so `revert_crop()` restores your exact hand-tuned margins at any time.
  - Supports **Coral EdgeTPU** (`/v1/projects/ambiled`), **Gemini Flash Vision** (`query_gemini_vision_cue`), or **offline pure-Python `I420` 92nd-percentile luma thresholding** (`estimate_i420_active_bbox_norm`).

### 3. Unbiased Bayer `bggr` Fringe Cancellation & Brown/Tan/Skin/Wood Protection
- **Symmetric Luma Weighting**: Inside `sample_box_fast()`, per-pixel weights use unbiased luma (`luma_f = yv - 8`) before averaging `Y, U, V`, so `+v` (red) and `-v` (cyan/green) Bayer demosaicing fringes along horizontal `2.35:1` letterbox boundaries cancel out to zero instead of biasing toward red.
- **Brown / Tan / Skin / Wood Protection**: RGB LEDs have no "brown" diode and turn dark wood, hair, leather, and skin blood-red unless green and blue are preserved. RPMLight distinguishes true vivid crimson (`ym >= 48, vm >= 24, um >= -5`) from warm brown/tan/skin tones and enforces a warm golden-amber floor (`g >= 0.72 * r, b >= 0.42 * r`).

### 4. Single-Slider Perceptual Curve (`1%–100%`) & Direct TV OSD Toasts
- A single dimmer slider (`1%–100%`) controls the entire perceptual engine via `apply_slider_curve(pct)`:
  - **`1%–25%` (Night Glow)**: Dimmer hardware cap (`12%–65%`), ultra-slow brightness glide, wide neighbor blending.
  - **`25%–50%` (Smooth Blend -> Balanced Sweet Spot)**: Smooth transitions with full Auto-Blackness Deletion.
  - **`50%–75%` (Crisp & Fast)**: Tighter outer-edge tracking and faster action response.
  - **`75%–100%` (Deep Vivid Rainbow)**: Full brightness, maximum speed, and deep inner-screen color reach.
- Includes `KodiJsonRpcToastSink` (`GUI.ShowNotification` over port `8080`) and Home Assistant `status_toast` publishing so adjusting the slider displays a short explanation right on the TV screen without switching HDMI inputs.

---

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from rpmlight import (
    CropCalibrator,
    CropMargins,
    KodiJsonRpcToastSink,
    NanoleafUdpSink,
    RPMLightEngine,
    TPUaaSBackgroundWorker,
    describe_slider_setting,
)

engine = RPMLightEngine(width=320, height=176, slider_pct=50)
sink = NanoleafUdpSink(ip="192.0.2.50", token="YOUR_NANOLEAF_TOKEN")
sink.activate_extcontrol(brightness=engine.nl_Master_brightness)

# Optional: preserve manual/hardcoded crop with opt-in Auto-Crop & 1-click revert
calibrator = CropCalibrator(CropMargins(top=328, bottom=216, left=608, right=352))

# Optional: low-rate (every 4s) Coral EdgeTPU scene-cue worker (disabled if base_url="")
tpu_worker = TPUaaSBackgroundWorker(base_url="")  # or set RPMLIGHT_TPU_URL
tpu_worker.start()

# Optional: direct local Kodi OSD toast notification (never changes HDMI input)
toast = KodiJsonRpcToastSink(host="192.0.2.60")
toast.send_toast(*describe_slider_setting(50, is_on=True))

# Inside your 15-20 FPS V4L2 / GStreamer I420 capture loop:
tpu_worker.offer_i420_frame(i420_frame_bytes)
if tpu_worker.latest_cue is not None:
    engine.apply_tpu_cue(tpu_worker.latest_cue)

result = engine.process_i420_frame(i420_frame_bytes)
sink.send_zones(result.zones_rgb, transition_ds=result.nl_transition_ds)
```

## License

MIT
