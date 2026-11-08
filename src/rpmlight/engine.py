"""Core Perceptual Multi-Zone RPMLight Ambilight Engine.

Implements five key perceptual & temporal innovations that eliminate the blinking,
strobing, and color-contamination problems of conventional camera/screen Ambilights:

1. **Decoupled Fast-Hue / Super-Slow-Brightness Tracking**:
   - Unit-normalized chromaticity (`ema_chroma`) updates immediately (`c_alpha = 0.64`)
     when colors move across the edge, while zone brightness (`ema_bri`) glides on a
     slow `~1.2s–1.8s` envelope unless a true full-screen Scene Cut (`>= 22` zones
     shifting brightness at once) is detected.
2. **Perceptual Constant-Lumen Wall Matching**:
   - Equalizes perceived wall brightness across 3-diode White (`W = 1.0`) vs. 1-diode
     Red/Blue (`W = 0.22..0.32`) via `W = 0.32*R + 0.46*G + 0.22*B` so rapid hue
     rotations never flash brighter or dimmer on the wall.
3. **Time-Dependent Falling Baseline with Schmitt Trigger, Neighbor-Rescue & Anti-Hotspot**:
   - When a zone first goes dark (e.g. a gray suit passing by), its brightness floor
     holds at the **Baseline Midpoint (`38.0`)** so nothing blinks off.
   - Only when a 5-zone neighborhood sits in sustained darkness does the floor slowly
     decay (`~9.5s`) down to **"On Just a Little Bit" (`14.0`)**, or `0.0` when the
     entire screen is pitch black.
   - **Schmitt Trigger** ignores slight camera flicker during decay, while **Neighbor
     Rescue** cascades dark spots back up to the Baseline whenever neighboring zones
     are lit, and **Anti-Hotspot Equalization** pulls lingering bright spots briskly
     down to the uniform Baseline before all zones dim together on black screens.
4. **Time-Depth Inner Fade-In (`zone_depth_mix`)**:
   - Tracks 100% pure outer-edge color whenever the edge is changing (`edge_delta >= 22`),
     and when the outer edge stops changing while a colorful region sits just inside the
     border (e.g. Polaroid collage frames or static stage borders), slowly fades in over
     `~3.5s` to a 50/50 average between the outer edge and inner color.
5. **Bayer `bggr` Fringe Cancellation, Subtitle Avoidance & Live-Action Shadow White-Mix**:
   - Neutralizes camera Bayer demosaicing teal/blue edge fringes, keeps bottom outer
     sampling boxes below 2-line TV subtitles, lifts the camera black pedestal (`ym - 6`),
     and blends warm white into shadowed colors so dark red-and-white flags stay
     red-and-white instead of turning black-and-red.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

DEFAULT_CROP_W = 320
DEFAULT_CROP_H = 176


@dataclass(frozen=True)
class ZoneFrameResult:
    """Computed 52-zone RGB output, transition time, scene-cut flag, and overhead sky."""

    zones_rgb: list[tuple[int, int, int]]
    nl_transition_ds: int
    is_scene_cut: bool
    up_rgb: list[int]
    up_bri: int
    down_rgb: list[int]
    down_bri: int


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
    """Precompute a tight Outer Edge Strip + a 36x42 Deep Inner Strip per zone."""
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
        # Avoid bottom-right slideshow clock overlays (x=282..315, y=154..170)
        if ex0 >= 265 and ey0 >= 142:
            ey0 = 138
        # Keep bottom-edge boxes at y=162..170 (below 2-line TV/Kodi subtitles at y=138..160)
        elif ey0 >= 152:
            ey0 = 162
            cur_eh = 8
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


def build_diamond_boxes(width: int = DEFAULT_CROP_W) -> dict[str, tuple[list[int], list[int]]]:
    """Precompute Top-33% Center-80% unified overhead ceiling sky sampling boxes."""
    return {
        "sky_left": _precompute_box_indices(width, 40, 8, 160, 56, step=12),
        "sky_right": _precompute_box_indices(width, 160, 8, 280, 56, step=12),
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
    """Extract perceptual RGB from an I420 sampling box with Bayer fringe suppression."""
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

        # Neutralize bggr sensor false teal/green offset ONLY on bright white/light-gray pixels (yv >= 85),
        # NEVER on dark/mid-tone navy-blue backgrounds (where yv=10..75 and u=+2..+25, v=-2..-20 is REAL blue!)
        if yv >= 85 and -8 <= u <= 24 and -24 <= v <= 4 and (u * u + v * v) < 750:
            u = int(u * 0.25)
            v = int(v * 0.25)
        elif 10 <= yv < 34 and u >= 2 and v <= -2:
            u = int(u * 1.55)
            v = int(v * 1.35)
        uv_sq = u * u + v * v

        vr = (v * 380) >> 8
        ub = (u * 320) >> 8
        chroma_lift = vr if vr > ub else ub
        if chroma_lift < 0:
            chroma_lift = 0
        peak_ch = yv + chroma_lift
        luma_sum += peak_ch

        luma_f = peak_ch - 8
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
                chroma_f = uv_eff - 110
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
    uvm_sq = um * um + vm * vm

    if ym >= 75:
        taper_limit = 220.0 if (um > 0 and vm <= 4) else 160.0
    else:
        taper_limit = 14.0
    if uvm_sq < taper_limit:
        c_taper = (uvm_sq / taper_limit) ** 1.2
        um = int(round(um * c_taper))
        vm = int(round(vm * c_taper))

    win_peak = ym + max(0, (vm * 380) >> 8, (um * 320) >> 8)
    if mean_luma < 20.0 and uvm_sq < 25:
        win_peak = min(win_peak, int(mean_luma * 1.75))

    pw = min(0.65, max(0.0, (mean_luma - 9.0) / 18.0) * 0.65)
    eff_luma = (1.0 - pw) * mean_luma + pw * float(win_peak)
    if eff_luma < 7.8:
        return (0, 0, 0)

    if eff_luma < 14.0:
        chroma_scale = max(0.35, (eff_luma - 7.8) / 6.2)
        um = int(round(um * chroma_scale))
        vm = int(round(vm * chroma_scale))

    c = ym - 6
    if c < 3:
        c = 3
    r = (298 * c + 409 * vm + 128) >> 8
    g = (298 * c - 100 * um - 208 * vm + 128) >> 8
    b = (298 * c + 516 * um + 128) >> 8
    r = 0 if r < 0 else (255 if r > 255 else r)
    g = 0 if g < 0 else (255 if g > 255 else g)
    b = 0 if b < 0 else (255 if b > 255 else b)

    if r >= 125 and g >= 38 and b >= 26 and r > g * 1.25 and b <= g * 1.35:
        b = min(255, max(b, int(round(g * 1.18))))

    rf = r / 255.0
    gf = g / 255.0
    bf = b / 255.0

    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    if s > 0.14:
        if h < 0.08 or h > 0.88:
            sat_boost = 1.05 if s < 0.72 else 1.20
        else:
            sat_boost = 1.32
        s = min(1.0, 0.14 + (s - 0.14) * sat_boost)
        # Live-Action Shadow White-Mix: in dark/shadowed regions (true luma ym < 48),
        # cap saturation to mix warm white into the color so shadowed flags stay red-and-white!
        if ym < 48:
            max_s = 0.58 + 0.42 * min(1.0, max(0.0, (ym - 10.0) / 38.0))
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
    r_gain = 242.0 - 12.0 * min(1.0, s / 0.20)
    g_gain = 240.0 + 5.0 * min(1.0, s / 0.20)
    b_gain = 236.0 + 19.0 * min(1.0, s / 0.20)
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
    """Stateful 52-zone Perceptual Zero-Blink Ambilight Processor."""

    def __init__(
        self,
        width: int = DEFAULT_CROP_W,
        height: int = DEFAULT_CROP_H,
        *,
        mode: str = "normal",
        blinkiness: int = 128,
    ) -> None:
        self.width = width
        self.height = height
        self.y_size = width * height
        self.uv_size = (width // 2) * (height // 2)
        self.mode = mode
        self.blinkiness = max(0, min(255, blinkiness))

        self.boxes_depth_glow = build_depth_glow_boxes(width, height)
        self.boxes_true_edge = build_sampling_boxes(width, height)
        self.diamond_boxes = build_diamond_boxes(width)
        self.num_zones = len(self.boxes_depth_glow)

        self.ema_chroma: list[list[float]] = [[1.0, 0.96, 0.90] for _ in range(self.num_zones)]
        self.ema_bri: list[float] = [38.0 for _ in range(self.num_zones)]
        self.zone_floor: list[float] = [38.0 for _ in range(self.num_zones)]
        self.was_screen_black: bool = False
        self.shared_black_floor: float = 36.0
        self.zone_depth_mix: list[float] = [0.0 for _ in range(self.num_zones)]
        self.prev_edge_rgb: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(self.num_zones)]
        self.cached_inner_rgb: list[tuple[int, int, int]] = [(0, 0, 0) for _ in range(self.num_zones)]
        self.sent_rgb: list[tuple[int, int, int]] = [(0, 0, 0) for _ in range(self.num_zones)]
        self.diamond_ema: dict[str, list[float]] = {
            "sky": [0.0, 0.0, 0.0],
            "sky_left": [0.0, 0.0, 0.0],
            "sky_right": [0.0, 0.0, 0.0],
        }
        self.scene_class: str = "auto"  # "auto" | "drama" | "music_video" (Coral TPU / classifier hook)
        self._frame_idx = 0
        self._frames_since_last_cut = 999

    def process_i420_frame(self, frame: bytes | memoryview) -> ZoneFrameResult:
        """Process a single cropped I420 frame and return all 52 zone colors + sky state."""
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
            raw_colors: list[tuple[int, int, int]] = []
            edge_deltas: list[float] = []
            for z_i, (_pid, ey_idx, euv_idx, iy_idx, iuv_idx) in enumerate(self.boxes_depth_glow):
                er, eg, eb = sample_box_fast(
                    y_plane, u_plane, v_plane, ey_idx, euv_idx, chroma_magnet=True
                )
                e_max = max(er, eg, eb)
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
                pe[0] += 0.35 * (er - pe[0])
                pe[1] += 0.35 * (eg - pe[1])
                pe[2] += 0.35 * (eb - pe[2])

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
                    or (e_max < 45 and i_sat >= 0.22)
                    or (e_max >= 120 and e_sat < 0.38 and i_sat >= 0.16)
                )
                if edge_delta >= 38.0:
                    self.zone_depth_mix[z_i] = max(0.0, self.zone_depth_mix[z_i] - 0.22)
                elif inner_has_color:
                    if is_bottom_zone:
                        max_depth = 0.55 if is_extra else 0.36
                        step_up = 0.009
                    elif is_distinct_pip or is_extra:
                        max_depth = 0.82
                        step_up = 0.055
                    else:
                        max_depth = 0.58
                        step_up = 0.018
                    self.zone_depth_mix[z_i] = min(max_depth, self.zone_depth_mix[z_i] + step_up)
                else:
                    self.zone_depth_mix[z_i] = max(0.0, self.zone_depth_mix[z_i] - 0.04)

                d_mix = self.zone_depth_mix[z_i]
                if d_mix <= 0.005:
                    raw_colors.append((er, eg, eb))
                else:
                    i_scale = 0.55 if (e_max < 40 and not is_extra) else 1.0
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

        if is_screen_black:
            if not self.was_screen_black:
                self.was_screen_black = True
                self.shared_black_floor = 36.0
                for i in range(num_zones):
                    self.zone_floor[i] = 36.0
            else:
                min_black_target = 0.0 if (global_mean_peak < 4.5 and max_raw_peak < 10) else 13.0
                self.shared_black_floor = max(min_black_target, self.shared_black_floor - 0.14)
                for i in range(num_zones):
                    self.zone_floor[i] = self.shared_black_floor
            target_bris = [max(self.shared_black_floor, float(raw_peaks[i])) for i in range(num_zones)]
        else:
            self.was_screen_black = False
            self.shared_black_floor = 36.0
            for i in range(num_zones):
                local_5_mean = sum(raw_peaks[(i + k) % num_zones] for k in range(-2, 3)) / 5.0
                local_13_mean = sum(raw_peaks[(i + k) % num_zones] for k in range(-6, 7)) / 13.0
                is_currently_sub = self.zone_floor[i] < midpoint_floor - 0.5

                if is_currently_sub:
                    if raw_peaks[i] >= 34.0 or edge_deltas[i] >= 42.0 or local_13_mean >= 32.0:
                        self.zone_floor[i] = midpoint_floor
                    else:
                        self.zone_floor[i] = max(14.0, self.zone_floor[i] - 0.14)
                else:
                    if raw_peaks[i] < 9.0 and local_5_mean < 11.0 and edge_deltas[i] < 16.0:
                        self.zone_floor[i] = max(14.0, midpoint_floor - 0.6)
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
        # Shot-Reverse-Shot Drama Protection: if cuts repeat within ~5s (90 frames), dampen brightness snap
        # unless scene_class is explicitly "music_video"
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

        for i in range(num_zones):
            prev_c = raw_colors[(i - 1) % num_zones]
            curr_c = raw_colors[i]
            next_c = raw_colors[(i + 1) % num_zones]

            sr = neigh_w * prev_c[0] + self_w * curr_c[0] + neigh_w * next_c[0]
            sg = neigh_w * prev_c[1] + self_w * curr_c[1] + neigh_w * next_c[1]
            sb = neigh_w * prev_c[2] + self_w * curr_c[2] + neigh_w * next_c[2]

            s_peak = max(sr, sg, sb)
            s_min = min(sr, sg, sb)
            s_sat = (s_peak - s_min) / s_peak if s_peak > 0.0 else 0.0

            if s_peak >= 4.0:
                tr = sr / s_peak
                tg = sg / s_peak
                tb = sb / s_peak
                if s_sat >= 0.22:
                    c_alpha = 0.24 + 0.70 * (b_norm**0.85)
                else:
                    c_alpha = 0.12 + 0.26 * b_norm
            else:
                tr, tg, tb = 0.98, 0.94, 0.88
                c_alpha = 0.045

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
            deadband = 5 if max(old_r, old_g, old_b) < 52 else 3
            if abs(ir - old_r) <= deadband and abs(ig - old_g) <= deadband and abs(ib - old_b) <= deadband:
                ir, ig, ib = old_r, old_g, old_b
            else:
                self.sent_rgb[i] = (ir, ig, ib)

        if is_scene_cut or self._frame_idx % 2 == 0 or self._frame_idx == 1:
            hue_alpha = 0.65 if is_scene_cut else (0.04 + 0.35 * (b_norm**1.5))
            lr, lg, lb = sample_box_fast(y_plane, u_plane, v_plane, *self.diamond_boxes["sky_left"])
            rr, rg, rb = sample_box_fast(y_plane, u_plane, v_plane, *self.diamond_boxes["sky_right"])
            for k, (dr, dg, db) in (
                ("sky_left", (lr, lg, lb)),
                ("sky_right", (rr, rg, rb)),
                ("sky", ((lr + rr) >> 1, (lg + rg) >> 1, (lb + rb) >> 1)),
            ):
                d_prev = self.diamond_ema[k]
                d_prev[0] += hue_alpha * (dr - d_prev[0])
                d_prev[1] += hue_alpha * (dg - d_prev[1])
                d_prev[2] += hue_alpha * (db - d_prev[2])

        hue_max_bri = int(round(90 + 95 * b_norm))
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
        color_diff = abs(l_rgb[0] - r_rgb[0]) + abs(l_rgb[1] - r_rgb[1]) + abs(l_rgb[2] - r_rgb[2])
        if color_diff > 240 and l_bri > 60 and r_bri > 60:
            u_rgb, u_bri = l_rgb, l_bri
            d_rgb, d_bri = r_rgb, r_bri
        else:
            u_rgb, u_bri = sky_rgb, sky_bri
            d_rgb, d_bri = sky_rgb, sky_bri

        return ZoneFrameResult(
            zones_rgb=list(self.sent_rgb),
            nl_transition_ds=nl_trans,
            is_scene_cut=is_scene_cut,
            up_rgb=u_rgb,
            up_bri=u_bri,
            down_rgb=d_rgb,
            down_bri=d_bri,
        )
