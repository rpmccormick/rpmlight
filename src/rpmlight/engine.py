"""Core Perceptual Multi-Zone RPMLight Ambilight Engine (v0.3.0).

Implements six key perceptual & temporal innovations that eliminate the blinking,
strobing, letterbox black-bar starvation, and red/dark-infection problems of
conventional camera/screen Ambilights:

1. **3-Step Orthogonal Auto-Blackness Deletion (2.35:1 Letterbox, Kodi Rows & 4:3 Pillarbox)**:
   - Each of the 52 perimeter zones precomputes 3 orthogonal inward march steps:
     - `Step 0`: True 16:9 outer edge (lifted off the bottom plastic bezel to `y = 155..164`).
     - `Step 1`: Moderate inward march (`14px` deeper vertically for top/bottom, `16px`
       horizontally for left/right).
     - `Step 2`: Deep inward march (`28px` deeper vertically to `y = 30..44` top / `y = 124..138`
       bottom for `2.35:1` widescreen movies & Kodi poster shelves; `32px` horizontally for
       `4:3` pillarbars).
   - Whenever an outer step reads black (`e_max < 26`), the zone automatically marches
     inward orthogonally until it hits real picture content, caching its active step for
     zero-overhead 20 FPS performance.
2. **Unbiased Bayer `bggr` Fringe Cancellation & Brown/Tan/Skin/Wood Protection**:
   - Uses symmetric per-pixel luma weighting (`luma_f = yv - 8`) inside `sample_box_fast()`
     so `+v` (red) and `-v` (cyan/green) Bayer demosaicing fringes along horizontal black
     bars cancel to zero instead of biasing toward red.
   - Distinguishes true vivid crimson (`ym >= 48, vm >= 24, um >= -5`) from warm brown,
     tan, wood, and skin tones so dark scenes never suffer from "red infection".
3. **Decoupled Fast-Hue / Super-Slow-Brightness Tracking**:
   - Unit-normalized chromaticity (`ema_chroma`) updates immediately when colors move
     across the edge, while zone brightness (`ema_bri`) glides on a slow envelope unless
     a full-screen Scene Cut (`>= 22` zones shifting brightness at once) is detected.
4. **Perceptual Constant-Lumen Wall Matching**:
   - Equalizes perceived wall brightness across 3-diode White (`W = 1.0`) vs. 1-diode
     Red/Blue (`W = 0.22..0.32`) via `W = 0.32*R + 0.46*G + 0.22*B`.
5. **Time-Dependent Falling Baseline with Anti-Dark-Infection Floor**:
   - Holds newly dark zones at the Baseline Midpoint (`38.0`), never drops below `24.0`
     while a movie is actively playing (eliminating dead black holes), and smoothly fades
     all zones together down to `0.0` when the entire screen is black.
6. **Single-Slider Perceptual Curve (`1%–100%`)**:
   - Maps a single dimmer slider across Night Glow (`1–25%`), Smooth Blend (`25–50%`),
     Balanced Sweet Spot (`50%`), Crisp & Fast Action (`50–75%`), and Deep Vivid Rainbow
     (`75–100%`).
"""

from __future__ import annotations

import colorsys
import math
import struct
from dataclasses import dataclass, field

DEFAULT_CROP_W = 320
DEFAULT_CROP_H = 176


def suppress_non_dominant(
    r: float, g: float, b: float, strength: float = 1.0
) -> tuple[float, float, float]:
    """Dominant-channel power-curve suppression scaled by strength (0.0 at 50% natural, 1.0 at 100% neon
    or pure crimson scenes) so Green diodes (~3x brighter to human eye) never wash out Red, Magenta, or Blue."""
    if strength <= 0.01:
        return (r, g, b)
    orig_r, orig_g, orig_b = r, g, b
    if r > g * 1.35 and r > b * 1.35 and r > 1.0:
        g = g * ((g / r) ** 2.0)
        b = b * ((b / r) ** 2.0)
    elif r > g * 1.45 and b > g * 1.25 and r > 1.0:
        g = g * ((g / r) ** 2.5)
    elif b > r * 1.25 and b > g * 1.15 and b > 1.0:
        r = r * ((r / b) ** 2.0)
        g = g * ((g / b) ** 1.6)
    return (
        orig_r + strength * (r - orig_r),
        orig_g + strength * (g - orig_g),
        orig_b + strength * (b - orig_b),
    )


class SubBassBeatDetector:
    """Pure-Python 3-Band Goertzel Sub-Bass (40-140 Hz) Spectral Flux Beat Detector for 16 kHz S16_LE PCM."""

    def __init__(self, sample_rate_hz: int = 16000, decimate: int = 4) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.decimate = decimate
        fs_d = sample_rate_hz / float(decimate)
        self._sub_coeffs = [2.0 * math.cos(2.0 * math.pi * f / fs_d) for f in (45.0, 65.0, 85.0, 105.0, 130.0)]
        self._mid_coeffs = [2.0 * math.cos(2.0 * math.pi * f / fs_d) for f in (190.0, 270.0)]
        self._hi_coeffs = [2.0 * math.cos(2.0 * math.pi * f / fs_d) for f in (1300.0, 1700.0)]
        self.sub_fast = 1.0
        self.sub_slow = 1.0
        self.rms_slow = 180.0
        self.beat_pulse = False
        self.beat_strength = 0.0
        self.audio_energy = 0.0
        self._cooldown_chunks = 0

    @staticmethod
    def _goertzel_power(samples: list[float], coeff: float) -> float:
        s1 = 0.0
        s2 = 0.0
        for x in samples:
            s0 = x + coeff * s1 - s2
            s2 = s1
            s1 = s0
        return max(0.0, s1 * s1 + s2 * s2 - coeff * s1 * s2)

    def process_pcm_s16le(self, pcm_bytes: bytes | memoryview) -> tuple[bool, float, float]:
        """Process a 50ms chunk (1600 bytes / 800 samples at 16 kHz) and return (beat_pulse, beat_strength, energy)."""
        n_samples = len(pcm_bytes) // 2
        if n_samples < self.decimate * 8:
            return (False, 0.0, 0.0)
        samples_all = struct.unpack(f"<{n_samples}h", pcm_bytes[: n_samples * 2])
        step = self.decimate
        dec = [
            sum(samples_all[idx : idx + step]) / float(step)
            for idx in range(0, n_samples - step + 1, step)
        ]
        n_dec = len(dec)
        mean_val = sum(dec) / float(n_dec)
        dec_zero = [x - mean_val for x in dec]
        rms = math.sqrt(sum(x * x for x in dec_zero) / float(n_dec))
        self.rms_slow = 0.96 * self.rms_slow + 0.04 * max(120.0, rms)

        norm = float(n_dec * n_dec)
        sub_pwr = max(self._goertzel_power(dec_zero, c) for c in self._sub_coeffs) / norm
        mid_pwr = max(self._goertzel_power(dec_zero, c) for c in self._mid_coeffs) / norm
        hi_pwr = max(self._goertzel_power(dec_zero, c) for c in self._hi_coeffs) / norm

        bass_combined = sub_pwr + 0.35 * mid_pwr + 0.10 * hi_pwr
        self.sub_fast = 0.55 * self.sub_fast + 0.45 * bass_combined
        self.sub_slow = 0.93 * self.sub_slow + 0.07 * max(25.0, bass_combined)
        flux_ratio = self.sub_fast / max(25.0, self.sub_slow)
        self.audio_energy = round(min(1.0, max(0.0, (rms - 130.0) / 1400.0)), 3)

        if self._cooldown_chunks > 0:
            self._cooldown_chunks -= 1

        is_kick = (
            rms >= max(240.0, self.rms_slow * 1.18)
            and sub_pwr >= 45.0
            and flux_ratio >= 1.65
            and self._cooldown_chunks == 0
        )
        if is_kick:
            self._cooldown_chunks = 3
            self.beat_pulse = True
            self.beat_strength = round(min(1.0, max(0.35, (flux_ratio - 1.35) / 1.85)), 3)
        else:
            self.beat_pulse = False
            self.beat_strength = round(max(0.0, self.beat_strength * 0.72), 3)
        return (self.beat_pulse, self.beat_strength, self.audio_energy)


@dataclass(frozen=True)
class ZoneFrameResult:
    """Computed 52-zone RGB output, transition time, scene-cut flag, and overhead/wing sky."""

    zones_rgb: list[tuple[int, int, int]]
    nl_transition_ds: int
    is_scene_cut: bool
    up_rgb: list[int]
    up_bri: int
    down_rgb: list[int]
    down_bri: int
    left_rgb: list[int] = field(default_factory=lambda: [255, 195, 145])
    left_bri: int = 32
    beat_pulse: bool = False
    energy: float = 0.0
    scene_class: str = "auto"
    is_video_paused: bool = False

    def to_diamond_dict(self, slider_pct: int = 50, lightshow_active: bool = True) -> dict[str, object]:
        """Build the MQTT diamond + lightshow sync payload for Home Assistant & TPUaaS."""
        return {
            "up_rgb": self.up_rgb,
            "up_bri": self.up_bri,
            "down_rgb": self.down_rgb,
            "down_bri": self.down_bri,
            "left_rgb": self.left_rgb,
            "left_bri": self.left_bri,
            "is_scene_cut": self.is_scene_cut,
            "beat_pulse": self.beat_pulse,
            "energy": self.energy,
            "scene_class": self.scene_class,
            "slider_pct": slider_pct,
            "lightshow_active": lightshow_active,
        }


@dataclass(frozen=True)
class SliderCurveParams:
    """Continuous perceptual parameters derived from a single 1-100% slider."""

    slider_pct: int
    blinkiness: int
    nl_Master_brightness: int
    edge_only_mix: float
    chroma_bleed_weight: float
    hue_max_bri: int


def apply_slider_curve(slider_pct: int) -> SliderCurveParams:
    """Map a single 1..100% user slider to RPMLight's continuous perceptual curve."""
    pct = max(1, min(100, int(slider_pct)))
    if pct <= 25:
        # 1%..25%: Night Glow -> Smooth Cinematic
        t = (pct - 1) / 24.0
        nl_bri = int(round(12 + t * 53))  # 12% -> 65%
        blinkiness = int(round(8 + t * 64))  # 8 -> 72
        edge_only_mix = 0.0
        chroma_bleed = 0.50 - t * 0.12  # 0.50 -> 0.38
        hue_max_bri = int(round(36 + t * 64))  # 36 -> 100
    elif pct <= 50:
        # 25%..50%: Smooth Cinematic -> Default 50% Sweet Spot
        t = (pct - 25) / 25.0
        nl_bri = int(round(65 + t * 20))  # 65% -> 85%
        blinkiness = int(round(72 + t * 68))  # 72 -> 140
        edge_only_mix = 0.0
        chroma_bleed = 0.38 - t * 0.16  # 0.38 -> 0.22
        hue_max_bri = int(round(100 + t * 40))  # 100 -> 140
    elif pct <= 75:
        # 50%..75%: Default Sweet Spot -> Crisp Pure Edge + Faster Action
        t = (pct - 50) / 25.0
        nl_bri = int(round(85 + t * 15))  # 85% -> 100%
        blinkiness = int(round(140 + t * 75))  # 140 -> 215
        edge_only_mix = t * 0.85  # 0.0 -> 0.85
        chroma_bleed = 0.22 * (1.0 - t)  # 0.22 -> 0.0
        hue_max_bri = int(round(140 + t * 45))  # 140 -> 185
    else:
        # 75%..100%: Crisp Pure Edge -> Deep Vivid Rainbow (deep inner color + max speed)
        t = (pct - 75) / 25.0
        nl_bri = 100
        blinkiness = int(round(215 + t * 40))  # 215 -> 255
        edge_only_mix = 0.85 * (1.0 - t)  # 0.85 -> 0.0 (re-enables deep inner color at 100%!)
        chroma_bleed = 0.0
        hue_max_bri = int(round(185 + t * 55))  # 185 -> 240
    return SliderCurveParams(
        slider_pct=pct,
        blinkiness=blinkiness,
        nl_Master_brightness=nl_bri,
        edge_only_mix=edge_only_mix,
        chroma_bleed_weight=chroma_bleed,
        hue_max_bri=hue_max_bri,
    )


def describe_slider_setting(slider_pct: int, is_on: bool = True) -> tuple[str, str]:
    """Return a concise (title, message) description for TV OSD toasts & HA status."""
    if not is_on:
        return (
            "TV AmbiLED: OFF",
            "Ambilight off — restoring warm white room lighting",
        )
    pct = max(1, min(100, int(slider_pct)))
    if pct <= 20:
        return (
            f"TV AmbiLED: {pct}% (Night Glow)",
            "Dim, ultra-relaxed ambient halo for dark-room viewing",
        )
    if pct <= 42:
        return (
            f"TV AmbiLED: {pct}% (Smooth Blend)",
            "Soft, dreamy color transitions with wide neighbor blending",
        )
    if pct <= 58:
        return (
            f"TV AmbiLED: {pct}% (Balanced Sweet Spot)",
            "Auto-crops black bars with zero-blink brightness & natural colors",
        )
    if pct <= 80:
        return (
            f"TV AmbiLED: {pct}% (Crisp & Fast)",
            "Tight edge tracking & snappy action response",
        )
    return (
        f"TV AmbiLED: {pct}% (Deep Vivid Rainbow)",
        "Max speed, full brightness & deep inner-screen color reach",
    )


def build_52_panel_coords() -> list[tuple[int, int, int]]:
    """Build the 52-zone clockwise perimeter coordinate ring (starting bottom-center)."""
    coords: list[tuple[int, int, int]] = []
    pid = 0
    for x in range(400, -1, -50):
        coords.append((pid, x, 0))
        pid += 1
    for y in range(50, 451, 50):
        coords.append((pid, 0, y))
        pid += 1
    for x in range(50, 851, 50):
        coords.append((pid, x, 450))
        pid += 1
    for y in range(400, -1, -50):
        coords.append((pid, 850, y))
        pid += 1
    for x in range(800, 449, -50):
        coords.append((pid, x, 0))
        pid += 1
    return coords


def _precompute_box_indices(
    width: int, x0: int, y0: int, x1: int, y1: int, step: int = 4
) -> tuple[list[int], list[int]]:
    uv_stride = width // 2
    y_indices: list[int] = []
    uv_indices: list[int] = []
    for py in range(y0, y1, step):
        y_row = py * width
        uv_row = (py >> 1) * uv_stride
        for px in range(x0, x1, step):
            y_indices.append(y_row + px)
            uv_indices.append(uv_row + (px >> 1))
    return y_indices, uv_indices


def build_sampling_boxes(
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
    x_min: int = 3,
    x_max: int = 315,
    y_min: int = 2,
    y_max: int = 170,
    box_w: int = 14,
    box_h: int = 12,
    step: int = 2,
) -> list[tuple[int, list[int], list[int]]]:
    """Precompute single-strip perimeter sampling boxes for each of the 52 zones."""
    span_x = (x_max - x_min) - box_w
    span_y = (y_max - y_min) - box_h
    precomputed: list[tuple[int, list[int], list[int]]] = []
    for pid, nx, ny in build_52_panel_coords():
        u = nx / 850.0
        v = 1.0 - (ny / 450.0)
        x0 = int(round(x_min + u * span_x))
        y0 = int(round(y_min + v * span_y))
        x1 = min(width, x0 + box_w)
        y1 = min(height, y0 + box_h)
        y_idx, uv_idx = _precompute_box_indices(width, x0, y0, x1, y1, step=step)
        precomputed.append((pid, y_idx, uv_idx))
    return precomputed


def build_depth_glow_boxes(
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
) -> list[tuple[int, list[int], list[int], list[int], list[int]]]:
    """Precompute a tight Outer Edge Strip + a Deep Inner Strip per zone."""
    ex_min, ex_max, ey_min, ey_max, ew, eh = 3, 315, 2, 170, 16, 14
    espan_x = (ex_max - ex_min) - ew
    espan_y = (ey_max - ey_min) - eh

    ix_min, ix_max, iy_min, iy_max, iw, ih = 18, 302, 18, 154, 54, 44
    ispan_x = (ix_max - ix_min) - iw
    ispan_y = (iy_max - iy_min) - ih

    precomputed: list[tuple[int, list[int], list[int], list[int], list[int]]] = []
    for pid, nx, ny in build_52_panel_coords():
        u = nx / 850.0
        v = 1.0 - (ny / 450.0)

        ex0 = int(round(ex_min + u * espan_x))
        ey0 = int(round(ey_min + v * espan_y))
        cur_eh = eh
        if ex0 >= 265 and ey0 >= 142:
            ey0 = 138
        elif ey0 >= 152:
            # Lift bottom edge off y=165..170 bezel reflection to y=155..164
            ey0 = 155
            cur_eh = 9
        ey_idx, euv_idx = _precompute_box_indices(
            width, ex0, ey0, min(width, ex0 + ew), min(height, ey0 + cur_eh), step=3
        )

        ix0 = int(round(ix_min + u * ispan_x))
        iy0 = int(round(iy_min + v * ispan_y))
        cur_ih = ih
        if ey0 >= 152:
            iy0 = 88
            cur_ih = 46
        iy_idx, iuv_idx = _precompute_box_indices(
            width, ix0, iy0, min(width, ix0 + iw), min(height, iy0 + cur_ih), step=8
        )

        precomputed.append((pid, ey_idx, euv_idx, iy_idx, iuv_idx))
    return precomputed


def build_auto_crop_depth_boxes(
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
) -> list[tuple[int, list[tuple[list[int], list[int]]], list[int], list[int]]]:
    """Precompute 3-Step Orthogonal Auto-Blackness Deletion Edge Strips + Deep Inner Strip.

    - Step 0: 16:9 full-screen edge (bottom lifted to y=155..164 to avoid bezel reflections).
    - Step 1: Moderate inward march (y=16..30 top, y=140..152 bottom, x+16 left, x-16 right).
    - Step 2: Deep inward march (y=30..44 top, y=124..138 bottom, x+32 left, x-32 right)
      to automatically skip 2.35:1 widescreen letterbox bars, Kodi inset poster shelves,
      and 4:3 pillarbox bars while preserving horizontal X (for top/bottom) and vertical Y
      (for left/right).
    """
    ex_min, ex_max, ey_min, ey_max, ew, eh = 3, 315, 2, 170, 16, 14
    espan_x = (ex_max - ex_min) - ew
    espan_y = (ey_max - ey_min) - eh

    ix_min, ix_max, iy_min, iy_max, iw, ih = 18, 302, 18, 154, 54, 44
    ispan_x = (ix_max - ix_min) - iw
    ispan_y = (iy_max - iy_min) - ih

    precomputed: list[tuple[int, list[tuple[list[int], list[int]]], list[int], list[int]]] = []
    for pid, nx, ny in build_52_panel_coords():
        u = nx / 850.0
        v = 1.0 - (ny / 450.0)

        ex0 = int(round(ex_min + u * espan_x))
        ey0 = int(round(ey_min + v * espan_y))

        edge_steps: list[tuple[list[int], list[int]]] = []
        if ny == 0:
            # Bottom edge (pid 0..8 and 44..51): march straight UP at the exact same X
            if ex0 >= 265:
                step_rects = [(ex0, 138, ew, 12), (ex0, 126, ew, 14), (ex0, 114, ew, 14)]
            else:
                step_rects = [(ex0, 155, ew, 9), (ex0, 140, ew, 12), (ex0, 124, ew, 14)]
        elif ny == 450:
            # Top edge (pid 18..34): march straight DOWN at the exact same X
            step_rects = [(ex0, 2, ew, 14), (ex0, 16, ew, 14), (ex0, 30, ew, 14)]
        elif nx == 0:
            # Left edge (pid 9..17): march straight RIGHT at the exact same Y (Step 2 = x=42..58 for 4:3)
            step_rects = [(3, ey0, ew, eh), (22, ey0, ew, eh), (42, ey0, ew, eh)]
        else:
            # Right edge (pid 35..43): march straight LEFT at the exact same Y (Step 2 = x=260..276 for 4:3)
            step_rects = [(ex0, ey0, ew, eh), (ex0 - 19, ey0, ew, eh), (ex0 - 39, ey0, ew, eh)]

        for sx0, sy0, sw, sh in step_rects:
            sx0_c = max(0, min(width - sw, sx0))
            sy0_c = max(0, min(height - sh, sy0))
            edge_steps.append(
                _precompute_box_indices(width, sx0_c, sy0_c, sx0_c + sw, sy0_c + sh, step=3)
            )

        ix0 = int(round(ix_min + u * ispan_x))
        iy0 = int(round(iy_min + v * ispan_y))
        cur_ih = ih
        if ny == 0:
            iy0 = 88
            cur_ih = 46
        iy_idx, iuv_idx = _precompute_box_indices(
            width, ix0, iy0, min(width, ix0 + iw), min(height, iy0 + cur_ih), step=8
        )

        precomputed.append((pid, edge_steps, iy_idx, iuv_idx))
    return precomputed


def build_diamond_boxes(width: int = DEFAULT_CROP_W) -> dict[str, tuple[list[int], list[int]]]:
    """Precompute overhead ceiling sky boxes + far-left wing box (for Kitchen Cabinet sync)."""
    return {
        "sky_left": _precompute_box_indices(width, 40, 8, 160, 56, step=12),
        "sky_right": _precompute_box_indices(width, 160, 8, 280, 56, step=12),
        "far_left": _precompute_box_indices(width, 8, 28, 72, 144, step=10),
    }


def sample_box_fast(
    y_plane: bytes | memoryview,
    u_plane: bytes | memoryview,
    v_plane: bytes | memoryview,
    y_indices: list[int],
    uv_indices: list[int],
    chroma_magnet: bool = True,
    ignore_bright_matte: bool = False,
) -> tuple[int, int, int]:
    """Extract perceptual RGB from an I420 box with symmetric Bayer cancellation & brown protection."""
    y_acc = 0
    u_acc = 0
    v_acc = 0
    w_acc = 0
    luma_sum = 0
    count = len(y_indices)

    for idx in range(count):
        yv = y_plane[y_indices[idx]]
        uv_i = uv_indices[idx]
        u = u_plane[uv_i] - 128
        v = v_plane[uv_i] - 128

        # Neutralize bggr sensor false teal/green offset ONLY on bright white/light-gray pixels (yv >= 85)
        if yv >= 85 and -8 <= u <= 24 and -24 <= v <= 4 and (u * u + v * v) < 750:
            u = int(u * 0.25)
            v = int(v * 0.25)
        elif 10 <= yv < 34 and u >= 2 and v <= -2:
            u = int(u * 1.55)
            v = int(v * 1.35)
        uv_sq = u * u + v * v

        # Symmetric per-pixel luma weighting: never bias weight by +v before averaging,
        # so +v (red) and -v (cyan/green) Bayer fringes along letterbox bars cancel to zero!
        luma_sum += yv
        luma_f = yv - 8
        if luma_f <= 0:
            w = 1
        else:
            if luma_f > 72:
                luma_f = 72
            if not chroma_magnet:
                w = 2 + (luma_f >> 2)
            else:
                if yv > 130 and uv_sq < 650:
                    md = 64 - (((yv - 130) * (650 - uv_sq)) >> 8)
                    min_md = 4 if ignore_bright_matte else 16
                    matte_damp = min_md if md < min_md else md
                else:
                    matte_damp = 64
                uv_eff = 550 if uv_sq > 550 else uv_sq
                chroma_f = uv_eff - 135
                if chroma_f < 0:
                    chroma_f = 0
                w = 4 + ((((luma_f >> 2) + ((chroma_f * luma_f) >> 9)) * matte_damp) >> 6)

        y_acc += yv * w
        u_acc += u * w
        v_acc += v * w
        w_acc += w

    if count == 0 or w_acc <= 0:
        return (0, 0, 0)

    mean_luma = luma_sum / count
    ym = y_acc // w_acc
    um = u_acc // w_acc
    vm = v_acc // w_acc

    # Suppress warm/red camera sensor noise in dark/shadowed regions (ym < 38)
    if ym < 38 and vm > 0 and vm < 18 and um >= -12:
        vm = int(vm * 0.35)
        um = int(um * 0.50)

    uvm_sq = um * um + vm * vm

    if ym >= 75:
        taper_limit = 220.0 if (um > 0 and vm <= 4) else 160.0
    else:
        taper_limit = 24.0
    if uvm_sq < taper_limit:
        c_taper = (uvm_sq / taper_limit) ** 1.2
        um = int(round(um * c_taper))
        vm = int(round(vm * c_taper))

    # Only allow +v (red) to lift perceived window peak if it is genuinely vivid red (vm >= 20)
    v_lift = ((vm * 300) >> 8) if vm >= 20 else 0
    u_lift = ((um * 320) >> 8) if um >= 12 else 0
    win_peak = ym + max(0, v_lift, u_lift)
    if mean_luma < 20.0 and uvm_sq < 25:
        win_peak = min(win_peak, int(mean_luma * 1.75))

    pw = min(0.65, max(0.0, (mean_luma - 9.0) / 18.0) * 0.65)
    eff_luma = (1.0 - pw) * mean_luma + pw * float(win_peak)
    if eff_luma < 8.5:
        return (0, 0, 0)

    if eff_luma < 16.0:
        chroma_scale = max(0.30, (eff_luma - 8.5) / 7.5)
        um = int(round(um * chroma_scale))
        vm = int(round(vm * chroma_scale))

    c = ym - 6
    if c < 3:
        c = 3

    # Brown / Tan / Skin / Wood protection:
    # RGB LEDs have no "brown" diode and turn brown/tan/wood blood-red unless G & B are preserved.
    is_vivid_red = ym >= 48 and vm >= 24 and um >= -5
    if vm > 0 and not is_vivid_red:
        r = (298 * c + 340 * vm + 128) >> 8
        g = (298 * c - 100 * um - 150 * vm + 128) >> 8
    else:
        r = (298 * c + 409 * vm + 128) >> 8
        g = (298 * c - 100 * um - 208 * vm + 128) >> 8
    b = (298 * c + 516 * um + 128) >> 8
    r = 0 if r < 0 else (255 if r > 255 else r)
    g = 0 if g < 0 else (255 if g > 255 else g)
    b = 0 if b < 0 else (255 if b > 255 else b)

    # Warm golden-tan floor for brown/wood/skin tones
    if r > g and vm > 0 and not is_vivid_red:
        min_g = int(round(r * 0.72))
        min_b = int(round(r * 0.42))
        if g < min_g:
            g = min_g
        if b < min_b:
            b = min_b

    if r >= 125 and g >= 38 and b >= 26 and r > g * 1.25 and b <= g * 1.35:
        b = min(255, max(b, int(round(g * 1.18))))

    rf = r / 255.0
    gf = g / 255.0
    bf = b / 255.0

    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    if s > 0.14:
        if h < 0.09 or h > 0.88:
            # Never over-boost weak red/brown hues into blood red
            sat_boost = 0.85 if s < 0.68 else 1.12
        else:
            sat_boost = 1.32
        s = min(1.0, 0.14 + (s - 0.14) * sat_boost)
        if ym < 52:
            max_s = 0.50 + 0.50 * min(1.0, max(0.0, (ym - 12.0) / 40.0))
            if s > max_s:
                s = max_s
    else:
        s = s * 0.25

    luma_drive = max(0.0, (eff_luma - 9.5) / 120.0)
    if eff_luma < 28.0:
        v_out = ((max(0.0, eff_luma - 9.5) / 18.5) ** 1.10) * 0.30
    else:
        v_drive = max(luma_drive, max(0.0, (v - 0.05) / 0.72))
        v_out = min(1.0, 0.30 + 0.70 * (max(0.0, (v_drive - 0.15) / 0.85) ** 0.72))
    if s > 0.22 and 0.50 <= h <= 0.72:
        v_out *= 0.54

    rf2, gf2, bf2 = colorsys.hsv_to_rgb(h, s, min(1.0, v_out))
    r_gain = 232.0 - 8.0 * min(1.0, s / 0.20)
    g_gain = 242.0 + 6.0 * min(1.0, s / 0.20)
    b_gain = 238.0 + 17.0 * min(1.0, s / 0.20)
    return (
        max(0, min(255, int(round(rf2 * r_gain)))),
        max(0, min(255, int(round(gf2 * g_gain)))),
        max(0, min(255, int(round(bf2 * b_gain)))),
    )


def hue_color_and_bri(rgb: tuple[int, int, int], max_bri: int = 140) -> tuple[list[int], int]:
    """Convert raw sky RGB into a vibrant Hue bulb RGB + gentle ambient brightness."""
    r, g, b = rgb
    peak = max(r, g, b)
    if peak < 12:
        return ([255, 195, 145], 32)
    h, s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if s > 0.14:
        s = min(1.0, 0.14 + (s - 0.14) * 1.55)
    else:
        s = s * 0.35
    rf, gf, bf = colorsys.hsv_to_rgb(h, s, 1.0)
    nr = min(255, int(round(rf * 235.0)))
    ng = min(255, int(round(gf * 250.0)))
    nb = min(255, int(round(bf * 255.0)))
    bri = max(32, min(max_bri, int(round(32 + (peak / 255.0) * (max_bri - 32)))))
    return ([nr, ng, nb], bri)


class RPMLightEngine:
    """Stateful 52-zone Perceptual Zero-Blink Ambilight Processor with Auto-Blackness Deletion."""

    def __init__(
        self,
        width: int = DEFAULT_CROP_W,
        height: int = DEFAULT_CROP_H,
        *,
        mode: str = "normal",
        blinkiness: int = 128,
        slider_pct: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.y_size = width * height
        self.uv_size = (width // 2) * (height // 2)
        self.mode = mode
        self.blinkiness = max(0, min(255, blinkiness))
        self.slider_pct = 50
        self.edge_only_mix = 0.0
        self.chroma_bleed_weight = 0.22
        self.nl_Master_brightness = 85
        self.hue_max_bri = int(round(90 + 95 * (self.blinkiness / 255.0)))
        if slider_pct is not None:
            self.set_slider_pct(slider_pct)

        self.boxes_depth_glow = build_depth_glow_boxes(width, height)
        self.boxes_auto_crop = build_auto_crop_depth_boxes(width, height)
        self.boxes_true_edge = build_sampling_boxes(width, height)
        self.diamond_boxes = build_diamond_boxes(width)
        self.num_zones = len(self.boxes_auto_crop)

        # Neutral warm-white initial chromaticity (never biases red on startup)
        self.ema_chroma: list[list[float]] = [[0.96, 0.96, 0.94] for _ in range(self.num_zones)]
        self.ema_bri: list[float] = [38.0 for _ in range(self.num_zones)]
        self.zone_floor: list[float] = [38.0 for _ in range(self.num_zones)]
        self.was_screen_black: bool = False
        self.shared_black_floor: float = 36.0
        self.zone_depth_mix: list[float] = [0.0 for _ in range(self.num_zones)]
        self.active_edge_step: list[int] = [0 for _ in range(self.num_zones)]
        self.prev_edge_rgb: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(self.num_zones)]
        self.cached_inner_rgb: list[tuple[int, int, int]] = [(0, 0, 0) for _ in range(self.num_zones)]
        self.sent_rgb: list[tuple[int, int, int]] = [(0, 0, 0) for _ in range(self.num_zones)]
        self.diamond_ema: dict[str, list[float]] = {
            "sky": [0.0, 0.0, 0.0],
            "sky_left": [0.0, 0.0, 0.0],
            "sky_right": [0.0, 0.0, 0.0],
            "far_left": [0.0, 0.0, 0.0],
        }
        self.scene_class: str = "auto"  # "auto" | "drama" | "music_video"
        self.wing_weights: dict[str, float] = {"west": 1.0, "east": 1.0, "far_left": 1.0}
        self._frame_idx = 0
        self._frames_since_last_cut = 999
        self.pause_steady_frames: int = 0
        self.is_video_paused: bool = False

    def apply_tpu_cue(self, cue: object) -> None:
        """Apply a low-rate TPUSceneCue (scene_class & wing_weights) from TPUaaS or Gemini."""
        s_class = getattr(cue, "scene_class", None)
        if s_class in ("auto", "drama", "music_video"):
            self.scene_class = str(s_class)
        ww = getattr(cue, "wing_weights", None)
        if isinstance(ww, dict):
            for k in ("west", "east", "far_left"):
                if k in ww:
                    target_w = max(0.50, min(1.50, float(ww[k])))
                    self.wing_weights[k] = round(0.55 * self.wing_weights[k] + 0.45 * target_w, 3)

    def set_slider_pct(self, slider_pct: int) -> SliderCurveParams:
        """Apply the single-slider 1..100% curve to this engine instance."""
        params = apply_slider_curve(slider_pct)
        self.slider_pct = params.slider_pct
        self.blinkiness = params.blinkiness
        self.nl_Master_brightness = params.nl_Master_brightness
        self.edge_only_mix = params.edge_only_mix
        self.chroma_bleed_weight = params.chroma_bleed_weight
        self.hue_max_bri = params.hue_max_bri
        return params

    def process_i420_frame(self, frame: bytes | memoryview) -> ZoneFrameResult:
        """Process a single cropped I420 frame and return all 52 zone colors + sky/wing state."""
        mv = memoryview(frame)
        y_plane = mv[0 : self.y_size]
        u_plane = mv[self.y_size : self.y_size + self.uv_size]
        v_plane = mv[self.y_size + self.uv_size : self.y_size + 2 * self.uv_size]

        self._frame_idx += 1
        b_norm = self.blinkiness / 255.0
        num_zones = self.num_zones

        if self.mode in ("normal", "depth_glow", "super_chill", "extra_color"):
            is_extra = self.mode == "extra_color"
            parity = self._frame_idx & 1
            probe_mod = self._frame_idx & 3
            raw_colors: list[tuple[int, int, int]] = []
            edge_deltas: list[float] = []
            for z_i, (_pid, edge_steps, iy_idx, iuv_idx) in enumerate(self.boxes_auto_crop):
                cur_step = self.active_edge_step[z_i]
                ey_idx, euv_idx = edge_steps[cur_step]
                er, eg, eb = sample_box_fast(
                    y_plane, u_plane, v_plane, ey_idx, euv_idx, chroma_magnet=True
                )
                e_max = max(er, eg, eb)

                # 3-Step Orthogonal Auto-Blackness Deletion:
                # Re-probe all 3 steps every 4th frame per zone OR immediately when a lit Step 0 goes black
                if (z_i & 3) == probe_mod or (
                    cur_step == 0 and e_max < 32 and max(self.prev_edge_rgb[z_i]) >= 24.0
                ):
                    s0_idx, s0_uv = edge_steps[0]
                    r0, g0, b0 = (
                        (er, eg, eb)
                        if cur_step == 0
                        else sample_box_fast(y_plane, u_plane, v_plane, s0_idx, s0_uv, chroma_magnet=True)
                    )
                    p0 = max(r0, g0, b0)
                    if p0 >= 32:
                        self.active_edge_step[z_i] = 0
                        er, eg, eb = r0, g0, b0
                        e_max = p0
                    else:
                        s1_idx, s1_uv = edge_steps[1]
                        r1, g1, b1 = (
                            (er, eg, eb)
                            if cur_step == 1
                            else sample_box_fast(y_plane, u_plane, v_plane, s1_idx, s1_uv, chroma_magnet=True)
                        )
                        s2_idx, s2_uv = edge_steps[2]
                        r2, g2, b2 = (
                            (er, eg, eb)
                            if cur_step == 2
                            else sample_box_fast(y_plane, u_plane, v_plane, s2_idx, s2_uv, chroma_magnet=True)
                        )
                        p1 = max(r1, g1, b1)
                        p2 = max(r2, g2, b2)
                        if p2 >= 26 and (p1 < 65 or p2 >= p1 * 1.35) and p2 >= p1 + 10:
                            self.active_edge_step[z_i] = 2
                            er, eg, eb = r2, g2, b2
                            e_max = p2
                        elif p1 >= 26:
                            self.active_edge_step[z_i] = 1
                            er, eg, eb = r1, g1, b1
                            e_max = p1
                        elif p2 >= p0 + 8:
                            self.active_edge_step[z_i] = 2
                            er, eg, eb = r2, g2, b2
                            e_max = p2
                        else:
                            self.active_edge_step[z_i] = 0
                            er, eg, eb = r0, g0, b0
                            e_max = p0

                e_min = min(er, eg, eb)
                e_sat = (e_max - e_min) / e_max if e_max > 0 else 0.0

                is_bottom_zone = z_i <= 6 or z_i >= 46
                if (z_i & 1) == parity or self._frame_idx == 1:
                    self.cached_inner_rgb[z_i] = sample_box_fast(
                        y_plane,
                        u_plane,
                        v_plane,
                        iy_idx,
                        iuv_idx,
                        chroma_magnet=True,
                        ignore_bright_matte=True,
                    )
                ir, ig, ib = self.cached_inner_rgb[z_i]
                i_max = max(ir, ig, ib)
                i_min = min(ir, ig, ib)
                i_sat = (i_max - i_min) / i_max if i_max > 0 else 0.0

                pe = self.prev_edge_rgb[z_i]
                edge_delta = abs(er - pe[0]) + abs(eg - pe[1]) + abs(eb - pe[2])
                edge_deltas.append(edge_delta)
                pe[0] += 0.55 * (er - pe[0])
                pe[1] += 0.55 * (eg - pe[1])
                pe[2] += 0.55 * (eb - pe[2])

                e_unit_r = er / e_max if e_max > 0 else 0.0
                e_unit_g = eg / e_max if e_max > 0 else 0.0
                e_unit_b = eb / e_max if e_max > 0 else 0.0
                i_unit_r = ir / i_max if i_max > 0 else 0.0
                i_unit_g = ig / i_max if i_max > 0 else 0.0
                i_unit_b = ib / i_max if i_max > 0 else 0.0
                unit_hue_dist = (
                    abs(i_unit_r - e_unit_r) + abs(i_unit_g - e_unit_g) + abs(i_unit_b - e_unit_b)
                )

                is_distinct_pip = (
                    (not is_bottom_zone) and i_max >= 24 and i_sat >= 0.22 and unit_hue_dist >= 0.48
                )
                inner_has_color = i_max >= 22 and (
                    is_distinct_pip
                    or (i_sat > e_sat + 0.08 and i_sat >= 0.18)
                    or (e_max < 65 and i_max >= e_max + 18)
                    or (e_max >= 120 and e_sat < 0.38 and i_sat >= 0.16)
                )
                if edge_delta >= 38.0:
                    self.zone_depth_mix[z_i] = max(0.0, self.zone_depth_mix[z_i] - 0.22)
                elif inner_has_color:
                    if is_bottom_zone:
                        max_depth = 0.65 if is_extra else (0.52 if e_max < 45 else 0.38)
                        step_up = 0.035 if e_max < 45 else 0.012
                    elif is_distinct_pip or is_extra:
                        max_depth = 0.85
                        step_up = 0.065
                    elif e_max < 55:
                        max_depth = 0.72
                        step_up = 0.045
                    else:
                        max_depth = 0.58
                        step_up = 0.018
                    self.zone_depth_mix[z_i] = min(max_depth, self.zone_depth_mix[z_i] + step_up)
                else:
                    self.zone_depth_mix[z_i] = max(0.0, self.zone_depth_mix[z_i] - 0.04)

                d_mix = self.zone_depth_mix[z_i] * (1.0 - self.edge_only_mix)
                if d_mix <= 0.005:
                    raw_colors.append((er, eg, eb))
                else:
                    i_scale = 0.80 if (e_max < 40 and not is_extra) else 1.0
                    raw_colors.append(
                        (
                            int(round((1.0 - d_mix) * er + d_mix * (ir * i_scale))),
                            int(round((1.0 - d_mix) * eg + d_mix * (ig * i_scale))),
                            int(round((1.0 - d_mix) * eb + d_mix * (ib * i_scale))),
                        )
                    )
        else:
            edge_deltas = [0.0 for _ in range(num_zones)]
            raw_colors = [
                sample_box_fast(y_plane, u_plane, v_plane, y_idx, uv_idx, chroma_magnet=False)
                for (_pid, y_idx, uv_idx) in self.boxes_true_edge
            ]

        neigh_w = 0.24 * (1.0 - 0.65 * b_norm)
        self_w = 1.0 - 2.0 * neigh_w

        raw_peaks = [max(c[0], c[1], c[2]) for c in raw_colors]
        global_mean_peak = sum(raw_peaks) / num_zones
        max_raw_peak = max(raw_peaks)
        is_screen_black = global_mean_peak < 11.0 and max_raw_peak < 32.0
        midpoint_floor = 38.0 if global_mean_peak >= 18.0 else max(26.0, global_mean_peak * 1.45)

        # Pause / Still-Frame Freeze Gate:
        # If average change across all 52 zones is < 2.5% for 10 consecutive frames (~0.5s),
        # lock the LED output buffer dead-still until a frame changes by > 4.0%!
        mean_delta_pct = (sum(edge_deltas) / (num_zones * 765.0)) * 100.0 if num_zones > 0 else 0.0
        max_delta_pct = (max(edge_deltas) / 765.0) * 100.0 if edge_deltas else 0.0
        if is_screen_black:
            self.pause_steady_frames = 0
            self.is_video_paused = False
        elif mean_delta_pct > 4.0 or max_delta_pct > 9.0:
            self.pause_steady_frames = 0
            self.is_video_paused = False
        elif mean_delta_pct < 2.5 and max_delta_pct < 6.5:
            self.pause_steady_frames += 1
            if self.pause_steady_frames >= 10:
                self.is_video_paused = True

        if is_screen_black:
            if not self.was_screen_black:
                self.was_screen_black = True
                self.shared_black_floor = 36.0
                for i in range(num_zones):
                    self.zone_floor[i] = 36.0
            else:
                self.shared_black_floor = max(0.0, self.shared_black_floor - 0.14)
                for i in range(num_zones):
                    self.zone_floor[i] = self.shared_black_floor
            target_bris = [self.shared_black_floor for _ in range(num_zones)]
        elif self.is_video_paused:
            target_bris = [self.ema_bri[i] for i in range(num_zones)]
        else:
            self.was_screen_black = False
            self.shared_black_floor = 36.0
            resting_min = 24.0
            for i in range(num_zones):
                local_5_mean = sum(raw_peaks[(i + k) % num_zones] for k in range(-2, 3)) / 5.0
                local_13_mean = sum(raw_peaks[(i + k) % num_zones] for k in range(-6, 7)) / 13.0
                is_currently_sub = self.zone_floor[i] < midpoint_floor - 0.5

                if is_currently_sub:
                    if raw_peaks[i] >= 34.0 or edge_deltas[i] >= 42.0 or local_13_mean >= 32.0:
                        self.zone_floor[i] = midpoint_floor
                    else:
                        self.zone_floor[i] = max(resting_min, self.zone_floor[i] - 0.10)
                else:
                    if raw_peaks[i] < 9.0 and local_5_mean < 11.0 and edge_deltas[i] < 16.0:
                        self.zone_floor[i] = max(resting_min, midpoint_floor - 0.6)
                    else:
                        self.zone_floor[i] = midpoint_floor

            target_bris = []
            for i in range(num_zones):
                n_m2 = (
                    self.zone_floor[(i - 2) % num_zones] >= midpoint_floor - 1.5
                    or raw_peaks[(i - 2) % num_zones] >= 28.0
                )
                n_m1 = (
                    self.zone_floor[(i - 1) % num_zones] >= midpoint_floor - 1.5
                    or raw_peaks[(i - 1) % num_zones] >= 28.0
                )
                n_p1 = (
                    self.zone_floor[(i + 1) % num_zones] >= midpoint_floor - 1.5
                    or raw_peaks[(i + 1) % num_zones] >= 28.0
                )
                n_p2 = (
                    self.zone_floor[(i + 2) % num_zones] >= midpoint_floor - 1.5
                    or raw_peaks[(i + 2) % num_zones] >= 28.0
                )
                lit_neighbors = int(n_m2) + int(n_m1) + int(n_p1) + int(n_p2)
                if (n_m1 and n_p1) or lit_neighbors >= 3:
                    self.zone_floor[i] = midpoint_floor
                else:
                    neigh_floor_avg = 0.5 * (
                        self.zone_floor[(i - 1) % num_zones] + self.zone_floor[(i + 1) % num_zones]
                    )
                    self.zone_floor[i] = max(self.zone_floor[i], neigh_floor_avg - 5.0)
                target_bris.append(max(self.zone_floor[i], float(raw_peaks[i])))

        cut_bri_zones = sum(1 for i in range(num_zones) if abs(target_bris[i] - self.ema_bri[i]) >= 55.0)
        mean_target_bri = sum(target_bris) / num_zones
        mean_ema_bri = sum(self.ema_bri) / num_zones
        raw_scene_cut = b_norm >= 0.25 and cut_bri_zones >= 22 and abs(mean_target_bri - mean_ema_bri) >= 28.0
        is_dialogue_ping_pong = (
            self.scene_class == "drama"
            or (self.scene_class == "auto" and self._frames_since_last_cut < 90)
        )
        if raw_scene_cut:
            self._frames_since_last_cut = 0
        else:
            self._frames_since_last_cut += 1
        is_scene_cut = raw_scene_cut and not is_dialogue_ping_pong
        nl_trans = 0 if is_scene_cut else (2 if b_norm < 0.30 else 1)

        # Anti-Virus Neutral Baseline: never let a single colored spot infect all dark zones
        uni_r = 0.96
        uni_g = 0.96
        uni_b = 0.94

        for i in range(num_zones):
            if self.is_video_paused and not is_screen_black:
                continue
            prev_c = raw_colors[(i - 1) % num_zones]
            curr_c = raw_colors[i]
            next_c = raw_colors[(i + 1) % num_zones]

            sr = neigh_w * prev_c[0] + self_w * curr_c[0] + neigh_w * next_c[0]
            sg = neigh_w * prev_c[1] + self_w * curr_c[1] + neigh_w * next_c[1]
            sb = neigh_w * prev_c[2] + self_w * curr_c[2] + neigh_w * next_c[2]

            s_peak = max(sr, sg, sb)
            s_min = min(sr, sg, sb)
            s_sat = (s_peak - s_min) / s_peak if s_peak > 0.0 else 0.0

            if (not is_screen_black) and s_peak >= 12.0:
                tr = sr / s_peak
                tg = sg / s_peak
                tb = sb / s_peak
                if s_peak < 28.0 and tr > tg + 0.04:
                    tr = tg + 0.04
                if s_sat >= 0.22:
                    c_alpha = 0.24 + 0.70 * (b_norm**0.85)
                else:
                    c_alpha = 0.12 + 0.26 * b_norm
            else:
                prev_ch = self.ema_chroma[(i - 1) % num_zones]
                next_ch = self.ema_chroma[(i + 1) % num_zones]
                bleed = self.chroma_bleed_weight
                tr = (1.0 - bleed) * uni_r + 0.5 * bleed * (prev_ch[0] + next_ch[0])
                tg = (1.0 - bleed) * uni_g + 0.5 * bleed * (prev_ch[1] + next_ch[1])
                tb = (1.0 - bleed) * uni_b + 0.5 * bleed * (prev_ch[2] + next_ch[2])
                c_alpha = 0.22 if is_screen_black else 0.08

            cr = self.ema_chroma[i][0] + c_alpha * (tr - self.ema_chroma[i][0])
            cg = self.ema_chroma[i][1] + c_alpha * (tg - self.ema_chroma[i][1])
            cb = self.ema_chroma[i][2] + c_alpha * (tb - self.ema_chroma[i][2])
            c_max = max(0.01, cr, cg, cb)
            self.ema_chroma[i][0] = cr / c_max
            self.ema_chroma[i][1] = cg / c_max
            self.ema_chroma[i][2] = cb / c_max

            t_bri = target_bris[i]
            p_bri = self.ema_bri[i]
            if is_scene_cut:
                b_alpha = 0.75
            elif p_bri < t_bri and t_bri <= midpoint_floor + 1.0:
                p_bri = max(p_bri, t_bri * 0.82)
                b_alpha = 0.42
            elif t_bri <= midpoint_floor + 1.0 and p_bri > midpoint_floor + 8.0:
                b_alpha = 0.22
            else:
                b_up = 0.025 + 0.080 * (b_norm**1.6)
                b_dn = 0.016 + 0.048 * (b_norm**1.6)
                b_alpha = b_up if t_bri > p_bri else b_dn

            new_bri = p_bri + b_alpha * (t_bri - p_bri)
            if is_screen_black and self.shared_black_floor <= 0.05 and new_bri < 1.2:
                new_bri = 0.0
            self.ema_bri[i] = new_bri

            w_lumen = (
                0.32 * self.ema_chroma[i][0]
                + 0.46 * self.ema_chroma[i][1]
                + 0.22 * self.ema_chroma[i][2]
            )
            lumen_match = 0.76 / (0.48 + 0.52 * w_lumen)
            eff_bri = new_bri * lumen_match

            ir = max(0, min(255, int(round(self.ema_chroma[i][0] * eff_bri))))
            ig = max(0, min(255, int(round(self.ema_chroma[i][1] * eff_bri))))
            ib = max(0, min(255, int(round(self.ema_chroma[i][2] * eff_bri))))

            old_r, old_g, old_b = self.sent_rgb[i]
            deadband = 0 if (is_screen_black or new_bri < 42.0) else 2
            if abs(ir - old_r) <= deadband and abs(ig - old_g) <= deadband and abs(ib - old_b) <= deadband:
                ir, ig, ib = old_r, old_g, old_b
            else:
                self.sent_rgb[i] = (ir, ig, ib)

        if is_scene_cut or self._frame_idx % 2 == 0 or self._frame_idx == 1:
            hue_alpha = 0.65 if is_scene_cut else (0.04 + 0.35 * (b_norm**1.5))
            lr, lg, lb = sample_box_fast(y_plane, u_plane, v_plane, *self.diamond_boxes["sky_left"])
            rr, rg, rb = sample_box_fast(y_plane, u_plane, v_plane, *self.diamond_boxes["sky_right"])
            flr, flg, flb = sample_box_fast(y_plane, u_plane, v_plane, *self.diamond_boxes["far_left"])
            for k, (dr, dg, db) in (
                ("sky_left", (lr, lg, lb)),
                ("sky_right", (rr, rg, rb)),
                ("sky", ((lr + rr) >> 1, (lg + rg) >> 1, (lb + rb) >> 1)),
                ("far_left", (flr, flg, flb)),
            ):
                d_prev = self.diamond_ema[k]
                d_prev[0] += hue_alpha * (dr - d_prev[0])
                d_prev[1] += hue_alpha * (dg - d_prev[1])
                d_prev[2] += hue_alpha * (db - d_prev[2])

        hue_max_bri = self.hue_max_bri
        sky_rgb, sky_bri = hue_color_and_bri(
            (
                int(round(self.diamond_ema["sky"][0])),
                int(round(self.diamond_ema["sky"][1])),
                int(round(self.diamond_ema["sky"][2])),
            ),
            max_bri=hue_max_bri,
        )
        l_rgb, l_bri = hue_color_and_bri(
            (
                int(round(self.diamond_ema["sky_left"][0])),
                int(round(self.diamond_ema["sky_left"][1])),
                int(round(self.diamond_ema["sky_left"][2])),
            ),
            max_bri=hue_max_bri,
        )
        r_rgb, r_bri = hue_color_and_bri(
            (
                int(round(self.diamond_ema["sky_right"][0])),
                int(round(self.diamond_ema["sky_right"][1])),
                int(round(self.diamond_ema["sky_right"][2])),
            ),
            max_bri=hue_max_bri,
        )
        fl_rgb, fl_bri = hue_color_and_bri(
            (
                int(round(self.diamond_ema["far_left"][0])),
                int(round(self.diamond_ema["far_left"][1])),
                int(round(self.diamond_ema["far_left"][2])),
            ),
            max_bri=max(28, int(round(hue_max_bri * 0.72))),
        )
        color_diff = abs(l_rgb[0] - r_rgb[0]) + abs(l_rgb[1] - r_rgb[1]) + abs(l_rgb[2] - r_rgb[2])
        if color_diff > 240 and l_bri > 60 and r_bri > 60:
            u_rgb, u_bri = l_rgb, l_bri
            d_rgb, d_bri = r_rgb, r_bri
        else:
            u_rgb, u_bri = sky_rgb, sky_bri
            d_rgb, d_bri = sky_rgb, sky_bri

        # Apply optional TPU/Gemini horizontal saliency wing weights (defaults to 1.0)
        u_bri = max(24, min(255, int(round(u_bri * self.wing_weights.get("west", 1.0)))))
        d_bri = max(24, min(255, int(round(d_bri * self.wing_weights.get("east", 1.0)))))
        fl_bri = max(24, min(255, int(round(fl_bri * self.wing_weights.get("far_left", 1.0)))))

        mean_edge_delta = sum(edge_deltas) / num_zones if num_zones > 0 else 0.0
        energy = round(
            min(
                1.0,
                (mean_edge_delta / 65.0) * 0.55
                + (global_mean_peak / 255.0) * 0.30
                + (0.25 if is_scene_cut else 0.0),
            ),
            3,
        )
        beat_pulse = bool(is_scene_cut or (mean_edge_delta >= 28.0 and cut_bri_zones >= 12))

        return ZoneFrameResult(
            zones_rgb=list(self.sent_rgb),
            nl_transition_ds=nl_trans,
            is_scene_cut=is_scene_cut,
            up_rgb=u_rgb,
            up_bri=u_bri,
            down_rgb=d_rgb,
            down_bri=d_bri,
            left_rgb=fl_rgb,
            left_bri=fl_bri,
            beat_pulse=beat_pulse,
            energy=energy,
            scene_class=self.scene_class,
            is_video_paused=self.is_video_paused,
        )
