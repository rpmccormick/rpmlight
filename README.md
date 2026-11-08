# RPMLight

**Perceptual Zero-Blink Camera & Screen Ambilight Engine**  
*Decoupled Fast-Hue / Super-Slow-Brightness · Perceptual Constant-Lumen Wall Matching · Schmitt-Trigger Falling Baseline · Time-Depth Inner Fade-In*

---

## Why RPMLight Exists

Traditional camera-based and screen-capture Ambilight systems (Nanoleaf 4D native camera mode, Hyperion, Govee Envisual) suffer from four fundamental problems on real live-action television:

1. **Blinky Live-Action Edges (Suits, Shadows & Motion):**  
   Cartoons work well on basic Ambilights because their edges are always brightly colored and only change hue. Live-action TV (monologues, news desks, dark suits, actors waving their arms in front of walls) constantly crosses black/shadow thresholds—causing individual LED zones to drop to `0` (off) and snap back on.
2. **White Glare vs. Single-Diode Color Imbalance:**  
   On RGB LED strips, `[255, 255, 255]` white lights all three diodes (`R+G+B`) and emits ~2.5× the total photon power of a pure primary color (`[255, 0, 0]` or `[0, 0, 255]`), causing a jarring brightness flash whenever an edge transitions between color and white/gray.
3. **Static Borders Hiding Inner Color (Polaroids, Collages & Picture-in-Picture Clips):**  
   Sampling only the outer 5% bezel misses Picture-in-Picture graphics (e.g. a red inset news clip inside a blue broadcast background) or Google Photos Polaroid collages with bright white borders. Sampling deep into the screen all the time ruins edge accuracy during motion.
4. **Camera Bayer Rainbow Aliasing & Subtitle Contamination:**  
   Raw `bggr` camera sensors produce severe cyan/blue/green demosaicing fringes on high-contrast white subtitles and sharp black-to-white edges.

---

## Core Architecture (Zero Settings Required)

RPMLight solves all four problems in a **100% Python stdlib, zero-dependency engine** (`RPMLightEngine`) capable of running directly on embedded ARM hardware (including the NXP i.MX6 inside a wall-mounted **Brilliant Control Panel** at ~18–20 FPS) or any Linux/Raspberry Pi host:

### 1. Decoupled Fast-Hue / Super-Slow-Brightness Tracking
Every zone separates color into **unit-normalized chromaticity (`ema_chroma`)** and **perceptual amplitude (`ema_bri`)**:
- **Hue / Chromaticity (`c_alpha = 0.64`)** updates **immediately** whenever a zone has color, so spinning portals or moving colors track with zero lag.
- **Brightness (`b_up ≈ 1.2s`, `b_dn ≈ 1.8s`)** glides on a super-slow envelope so passing shadows or arm waves never blink—**except** on a true full-screen **Scene Cut** (`>= 22` of 52 zones shifting brightness by `>= 55` simultaneously), which snaps immediately.
- **Shot-Reverse-Shot Drama Protection (with Coral EdgeTPU / Classifier Hook):** Repeated cuts within 5 seconds (e.g. dialogue shot-reverse-shot cutting back and forth between two camera lightings) automatically dampen brightness jumps while keeping hue responsive.

### 2. Perceptual Constant-Lumen Wall Matching
RPMLight models total perceived wall lumen output across all three RGB LED channels:
$$W(R, G, B) = 0.32 R + 0.46 G + 0.22 B$$
and scales zone amplitude by:
$$\text{LumenMatch} = \frac{0.76}{0.48 + 0.52 \cdot W(R, G, B)}$$
so switching rapidly between 3-diode White and 1-diode Red/Blue maintains constant perceived room brightness.

### 3. Time-Dependent Falling Baseline + Schmitt Trigger + Neighbor-Rescue Cascade
- **Baseline Midpoint (`38.0`):** When a zone first turns dark (an actor in a gray suit walking past, or a cut to a dark monologue stage), its floor holds at the **Baseline Midpoint (`38.0`)** so dark objects never cause a drop-out blink.
- **Slow Sustained-Dark Decay (`38.0 -> 14.0` over ~9.5s):** Only when a zone **and** its 5-zone neighborhood sit in continuous darkness (such as the dark bottom-left/right corners of a long monologue shot) does its floor slowly melt down to **"On Just a Little Bit" (`14.0`)**, and only reaches `0.0` when the entire screen is pitch black.
- **Schmitt Trigger & Neighbor-Rescue Cascade:** Once decaying below baseline, slight camera noise is ignored until a drastic change (`raw_peaks >= 34` or `edge_delta >= 42`) occurs—or until neighboring zones on both sides go to baseline, which triggers a smooth **Neighbor-Rescue Cascade** so isolated dark holes can never form.
- **Anti-Hotspot Equalization on Black Screens:** When the screen goes dark/paused, lingering bright hot spots above the baseline are pulled down briskly (`b_alpha = 0.22`) to equalize at the `36–38` baseline before all 52 zones slowly dim down together in unison.

### 4. Time-Depth Inner Fade-In (`zone_depth_mix`)
Each zone precomputes both a razor-thin **Outer Edge Strip (`16×14`)** and a **Deep Inner Strip (`54×44`)**:
- Whenever the outer edge is actively changing (`edge_delta >= 38`), `zone_depth_mix` resets toward `0.0` (**100% pure outer-edge accuracy**).
- When the outer edge is static/boring (a solid broadcast background, Polaroid frame, or dark border) and the inner strip contains a colorful feature or distinct Picture-in-Picture clip (`is_distinct_pip`), `zone_depth_mix` smoothly fades in over time (`up to 82%` for distinct PiP clips, `58%` for general inner color) so inset clips and collage photos bloom into the outline.

### 5. Bayer `bggr` Fringe Suppression, Subtitle Avoidance & Shadow White-Mix
- Bottom outer sampling boxes sit at `y=162..170`—completely below 2-line TV/Kodi subtitles (`y=138..160`).
- Camera black pedestal is calibrated (`c = max(3, ym - 6)`) so dark gray suits keep positive luma instead of crushing to `[0, 0, 0]` at BT.601 `ym = 16`.
- In shadowed regions (`eff_luma < 46`), maximum HSV saturation is capped to mix warm white into dark colors—keeping shadowed red-and-white flags looking red-and-white instead of black-and-red.

---

## Installation & Quick Start

```bash
pip install .
```

### Run on a Brilliant Control Panel (NXP i.MX6 Hardware ISP -> Nanoleaf 4D + MQTT Sky Hue)
```bash
rpmlight \
  --source brilliant \
  --device /dev/video0 \
  --crop-x 344 --crop-y 108 --width 320 --height 184 \
  --nanoleaf-ip 192.168.10.78 \
  --nanoleaf-token YOUR_NANOLEAF_TOKEN \
  --mqtt-host 192.168.10.10 --mqtt-user rpm --mqtt-pass 111
```

### Run on Linux / Raspberry Pi with a V4L2 Webcam -> WLED or Nanoleaf 4D
```bash
rpmlight \
  --source v4l2 \
  --device /dev/video0 \
  --wled-ip 192.168.10.150
```

---

## Roadmap
- **Coral EdgeTPU / Scene Genre Classification (`scene_class`):** Real-time classification between high-energy music videos/concerts (`scene_class="music_video"`) vs. narrative drama/dialogue (`scene_class="drama"`) to dynamically adapt scene-cut responsiveness.
- **Built-in `brilliant-mqtt` Integration:** Native on-panel deployment and Home Assistant MQTT Discovery controls in [`brilliant-mqtt`](https://github.com/joyfulhouse/brilliant-mqtt).

## License
MIT
