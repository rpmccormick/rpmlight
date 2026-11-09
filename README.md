# RPMLight (`rpmlight`) — v0.3.0

**Perceptual Zero-Blink Multi-Zone Camera & Screen Ambilight Engine for Nanoleaf 4D, WLED, and Home Assistant Hue Sky / Wing Lights.**

Traditional camera and screen-capture Ambilights (Hyperion, SignalRGB, Prismatik, stock Nanoleaf 4D) treat every zone as a raw RGB pixel mirror. In real living rooms, that causes **constant distracting brightness strobing**, **black-bar starvation on `2.35:1` widescreen movies & Kodi poster menus**, **blood-red skin/wood/shadow contamination**, and **flickering when actors walk across the frame**.

**RPMLight** solves these problems using a human-vision-inspired perceptual pipeline designed to run at **15–20 FPS in pure Python with <10% CPU usage** on embedded Linux / Android wall switches (such as the Brilliant Smart Home Control) or Raspberry Pi.

---

## Key Innovations (v0.3.0)

### 1. 3-Step Orthogonal Auto-Blackness Deletion (`2.35:1` Letterbox, Kodi Rows & `4:3` Pillarbox)
- Each of the 52 perimeter zones precomputes **3 orthogonal inward march steps**:
  - **`Step 0` (16:9 Outer Edge)**: Samples the true 16:9 perimeter (with bottom zones lifted to `y = 155..164` off the bottom TV bezel reflection).
  - **`Step 1` (Moderate Inward March)**: Steps `14px` deeper vertically on top/bottom (`y = 16..30` / `y = 140..152`) or `16px` deeper horizontally on left/right (`x = 19..35` / `x = 283..299`).
  - **`Step 2` (Deep Inward March)**: Steps `28px` deeper vertically (`y = 30..44` top / `y = 124..138` bottom) or `32px` deeper horizontally (`x = 35..51` left / `x = 267..283` right).
- Whenever an outer step reads black (`e_max < 26`), the zone automatically steps inward **orthogonally** (preserving exact horizontal `X` for top/bottom zones and exact vertical `Y` for left/right zones) so `2.35:1` widescreen movies, Kodi inset poster shelves, and `4:3` pillarboxed shows light up every zone at the default `50%` setting.

### 2. Unbiased Bayer `bggr` Fringe Cancellation & Brown/Tan/Skin/Wood Protection
- **Symmetric Luma Weighting**: Inside `sample_box_fast()`, per-pixel weights use unbiased luma (`luma_f = yv - 8`) before averaging `Y, U, V`, so `+v` (red) and `-v` (cyan/green) Bayer demosaicing fringes along horizontal `2.35:1` letterbox boundaries cancel out to zero instead of biasing toward red.
- **Brown / Tan / Skin / Wood Protection**: RGB LEDs have no "brown" diode and turn dark wood, hair, leather, and skin blood-red unless green and blue are preserved. RPMLight distinguishes true vivid crimson (`ym >= 48, vm >= 24, um >= -5`) from warm brown/tan/skin tones and enforces a warm golden-amber floor (`g >= 0.72 * r, b >= 0.42 * r`).

### 3. Single-Slider Perceptual Curve (`1%–100%`) & Direct TV OSD Toasts
- A single dimmer slider (`1%–100%`) controls the entire perceptual engine via `apply_slider_curve(pct)`:
  - **`1%–25%` (Night Glow)**: Dimmer hardware cap (`12%–65%`), ultra-slow brightness glide, wide neighbor blending.
  - **`25%–50%` (Smooth Blend -> Balanced Sweet Spot)**: Smooth transitions with full Auto-Blackness Deletion.
  - **`50%–75%` (Crisp & Fast)**: Tighter outer-edge tracking and faster action response.
  - **`75%–100%` (Deep Vivid Rainbow)**: Full brightness, maximum speed, and deep inner-screen color reach.
- Includes `KodiJsonRpcToastSink` (`GUI.ShowNotification` over port `8080`) and Home Assistant `status_toast` publishing so adjusting the slider displays a short explanation right on the TV screen without switching HDMI inputs.

### 4. Decoupled Fast-Hue / Super-Slow-Brightness Tracking & Shot-Reverse-Shot Drama Protection
- **Fast Chromaticity (`ema_chroma`)**: Unit-normalized RGB ratios (`max(r, g, b) = 1.0`) update rapidly (`c_alpha = 0.64`) so moving objects and panning colors track smoothly around the perimeter.
- **Super-Slow Brightness (`ema_bri`)**: Brightness glides on a slow `~1.2s–1.8s` envelope during normal viewing, snapping (`b_alpha = 0.75`) only when a true full-screen **Scene Cut** (`>= 22` of 52 zones shifting brightness by `>= 55` simultaneously) is detected.
- **Shot-Reverse-Shot Dialogue Protection**: If scene cuts repeat within `~5 seconds` (`< 90` frames, e.g. back-and-forth dialogue between two actors with different background lighting), brightness snapping is automatically dampened so dialogue scenes never pump the room lights.

### 5. Perceptual Constant-Lumen Wall Matching
- A raw RGB LED at `[255, 255, 255]` fires all 3 diodes (`3.0x` total light), whereas `[255, 0, 0]` or `[0, 0, 255]` fires only 1 diode—causing white/yellow passages to flash `3x` brighter on the wall than red/blue passages at the same numerical brightness.
- RPMLight normalizes output brightness by perceived wall luminance (`W = 0.32*R + 0.46*G + 0.22*B`, `lumen_match = 0.76 / (0.48 + 0.52 * W)`), keeping wall illumination constant as hues rotate.

### 6. Time-Dependent Falling Baseline with Anti-Dark-Infection Floor
- **Baseline Midpoint (`38.0`)**: When a zone first goes dark (e.g. a dark suit passing by), its brightness floor holds at `38.0` so nothing blinks off.
- **Anti-Dark-Infection Floor (`24.0`)**: During active playback (`not is_screen_black`), no zone ever drops below `24.0` or gets infected by a neighboring red spot (`uni_r/g/b = 0.96, 0.96, 0.94`), and when the entire screen turns pitch black (`is_screen_black`), all 52 zones equalize to `36.0` and fade smoothly down to `0.0` together.

---

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from rpmlight import KodiJsonRpcToastSink, NanoleafUdpSink, RPMLightEngine, describe_slider_setting

engine = RPMLightEngine(width=320, height=176, slider_pct=50)
sink = NanoleafUdpSink(ip="192.168.1.50", token="YOUR_NANOLEAF_TOKEN")
sink.activate_extcontrol(brightness=engine.nl_Master_brightness)

# Optional direct local Kodi OSD toast notification (never changes HDMI input)
toast = KodiJsonRpcToastSink(host="192.168.100.11")
toast.send_toast(*describe_slider_setting(50, is_on=True))

# Inside your 15-20 FPS V4L2 / GStreamer I420 capture loop:
result = engine.process_i420_frame(i420_frame_bytes)
sink.send_zones(result.zones_rgb, transition_ds=result.nl_transition_ds)
```

## License

MIT
