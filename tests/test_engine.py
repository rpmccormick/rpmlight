"""Unit tests for the Perceptual Multi-Zone RPMLight Engine (v0.3.0)."""

from __future__ import annotations

from rpmlight.engine import (
    DEFAULT_CROP_H,
    DEFAULT_CROP_W,
    RPMLightEngine,
    apply_slider_curve,
    build_52_panel_coords,
    build_auto_crop_depth_boxes,
    build_depth_glow_boxes,
    describe_slider_setting,
    sample_box_fast,
)
from rpmlight.sinks import NanoleafUdpSink, WledUdpSink


def _make_i420_frame(
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
    y_val: int = 16,
    u_val: int = 128,
    v_val: int = 128,
) -> bytearray:
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    buf = bytearray(y_size + 2 * uv_size)
    buf[0:y_size] = bytes([y_val]) * y_size
    buf[y_size : y_size + uv_size] = bytes([u_val]) * uv_size
    buf[y_size + uv_size : y_size + 2 * uv_size] = bytes([v_val]) * uv_size
    return buf


def _paint_i420_rect(
    buf: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    y_val: int,
    u_val: int,
    v_val: int,
    width: int = DEFAULT_CROP_W,
    height: int = DEFAULT_CROP_H,
) -> None:
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    uv_stride = width // 2
    for py in range(max(0, y0), min(height, y1)):
        row = py * width
        uv_row = (py >> 1) * uv_stride
        for px in range(max(0, x0), min(width, x1)):
            buf[row + px] = y_val
            uv_idx = uv_row + (px >> 1)
            buf[y_size + uv_idx] = u_val
            buf[y_size + uv_size + uv_idx] = v_val


def test_52_panel_coords_count() -> None:
    coords = build_52_panel_coords()
    assert len(coords) == 52
    assert coords[0] == (0, 400, 0)
    auto_boxes = build_auto_crop_depth_boxes()
    assert len(auto_boxes) == 52
    for _pid, edge_steps, _iy, _iuv in auto_boxes:
        assert len(edge_steps) == 3


def test_sample_box_fast_preserves_dark_gray_suit_and_red_white_flag() -> None:
    # Dark gray suit (yv=13, neutral u=128, v=128) stays above cutoff (not crushed at ym=16)
    frame_gray = _make_i420_frame(y_val=13, u_val=128, v_val=128)
    boxes = build_depth_glow_boxes()
    _pid, ey_idx, euv_idx, _iy_idx, _iuv_idx = boxes[0]
    y_size = DEFAULT_CROP_W * DEFAULT_CROP_H
    uv_size = (DEFAULT_CROP_W // 2) * (DEFAULT_CROP_H // 2)
    mv = memoryview(frame_gray)
    rgb_gray = sample_box_fast(
        mv[:y_size], mv[y_size : y_size + uv_size], mv[y_size + uv_size :], ey_idx, euv_idx
    )
    assert max(rgb_gray) > 0

    # Shadowed red flag (yv=22, v=165) mixes warm white (g > 0, b > 0) instead of pure [R, 0, 0]
    frame_flag = _make_i420_frame(y_val=22, u_val=120, v_val=165)
    mv_flag = memoryview(frame_flag)
    r, g, b = sample_box_fast(
        mv_flag[:y_size],
        mv_flag[y_size : y_size + uv_size],
        mv_flag[y_size + uv_size :],
        ey_idx,
        euv_idx,
    )
    assert r >= g > 0
    assert b > 0


def test_brown_skin_wood_protection_and_bayer_fringe_cancellation() -> None:
    boxes = build_depth_glow_boxes()
    _pid, ey_idx, euv_idx, _iy_idx, _iuv_idx = boxes[0]
    y_size = DEFAULT_CROP_W * DEFAULT_CROP_H
    uv_size = (DEFAULT_CROP_W // 2) * (DEFAULT_CROP_H // 2)

    # Warm brown/wood/skin (ym=34, u=116, v=144) must stay warm golden-tan, not blood-red
    frame_brown = _make_i420_frame(y_val=34, u_val=116, v_val=144)
    mv_b = memoryview(frame_brown)
    r_b, g_b, b_b = sample_box_fast(
        mv_b[:y_size], mv_b[y_size : y_size + uv_size], mv_b[y_size + uv_size :], ey_idx, euv_idx
    )
    assert g_b >= int(r_b * 0.55)
    assert b_b > 0

    # Symmetric +v / -v Bayer demosaicing fringe at equal luma cancels to neutral (no red bias)
    frame_fringe = _make_i420_frame(y_val=28, u_val=128, v_val=128)
    for idx, uv_i in enumerate(euv_idx):
        frame_fringe[y_size + uv_size + uv_i] = 148 if (idx % 2 == 0) else 108
    mv_f = memoryview(frame_fringe)
    r_f, g_f, _b_f = sample_box_fast(
        mv_f[:y_size], mv_f[y_size : y_size + uv_size], mv_f[y_size + uv_size :], ey_idx, euv_idx
    )
    assert abs(r_f - g_f) <= 22


def test_auto_blackness_deletion_top_bottom_widescreen_and_kodi_rows() -> None:
    # Deep 2.35:1 widescreen / Kodi inset poster layout:
    # Top (y=0..30) & Bottom (y=140..176) are black (Y=4), while y=30..88 is vivid blue and y=88..140 is vivid green
    frame = _make_i420_frame(y_val=4, u_val=128, v_val=128)
    _paint_i420_rect(frame, 0, 30, DEFAULT_CROP_W, 88, y_val=115, u_val=215, v_val=110)
    _paint_i420_rect(frame, 0, 88, DEFAULT_CROP_W, 140, y_val=115, u_val=75, v_val=75)

    engine = RPMLightEngine(slider_pct=50)
    for _ in range(4):
        res = engine.process_i420_frame(frame)

    # Top zones (z_i=22..30) should step down to Step 2 and render vivid blue
    for z_i in range(22, 31):
        assert engine.active_edge_step[z_i] == 2
        r, _g, b = res.zones_rgb[z_i]
        assert b > r + 15

    # Bottom zones (z_i=1..6) should step up to Step 2 and render vivid green
    for z_i in range(1, 7):
        assert engine.active_edge_step[z_i] == 2
        r, g, _b = res.zones_rgb[z_i]
        assert g > r + 15


def test_auto_blackness_deletion_left_right_4x3_pillarbox() -> None:
    # True 4:3 pillarbox in 320x176: left (x=0..42) and right (x=278..320) are black (Y=4), center is vivid blue
    frame = _make_i420_frame(y_val=4, u_val=128, v_val=128)
    _paint_i420_rect(frame, 42, 0, 278, DEFAULT_CROP_H, y_val=120, u_val=215, v_val=110)

    engine = RPMLightEngine(slider_pct=50)
    for _ in range(4):
        res = engine.process_i420_frame(frame)

    # Left edge (z_i=11..15) and right edge (z_i=37..41) should step inward horizontally to Step 2
    for z_i in (11, 13, 15, 37, 39, 41):
        assert engine.active_edge_step[z_i] == 2
        r, _g, b = res.zones_rgb[z_i]
        assert b > r + 15


def test_single_slider_curve_and_descriptions() -> None:
    p10 = apply_slider_curve(10)
    p50 = apply_slider_curve(50)
    p70 = apply_slider_curve(70)
    p100 = apply_slider_curve(100)
    assert p10.blinkiness < p50.blinkiness < p70.blinkiness < p100.blinkiness
    assert p70.edge_only_mix > p50.edge_only_mix
    assert p100.edge_only_mix == 0.0

    title_50, msg_50 = describe_slider_setting(50, is_on=True)
    assert "50%" in title_50
    assert "black bars" in msg_50
    title_off, _msg_off = describe_slider_setting(50, is_on=False)
    assert "OFF" in title_off


def test_falling_baseline_and_equalized_black_screen() -> None:
    engine = RPMLightEngine()
    bright_red = _make_i420_frame(y_val=110, u_val=100, v_val=200)
    res1 = engine.process_i420_frame(bright_red)
    assert len(res1.zones_rgb) == 52
    assert res1.zones_rgb[0][0] > res1.zones_rgb[0][2]

    dark_frame = _make_i420_frame(y_val=6, u_val=128, v_val=128)
    res_dark1 = engine.process_i420_frame(dark_frame)
    peaks_first = [max(c) for c in res_dark1.zones_rgb]
    assert min(peaks_first) >= 24

    for _ in range(120):
        res_dark_later = engine.process_i420_frame(dark_frame)
    peaks_later = [max(c) for c in res_dark_later.zones_rgb]
    assert max(peaks_later) < min(peaks_first)


def test_sinks_packet_encoding() -> None:
    zones = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    nl_pkt = NanoleafUdpSink.encode_v2_packet(zones, transition_ds=1)
    assert len(nl_pkt) == 2 + 3 * 8
    wled = WledUdpSink("127.0.0.1")
    wled_pkt = wled.encode_drgb_packet(zones)
    assert wled_pkt[:2] == b"\x02\x02"
    assert len(wled_pkt) == 2 + 3 * 3
