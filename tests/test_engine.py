"""Unit tests for the Perceptual Multi-Zone RPMLight Engine."""

from __future__ import annotations

from rpmlight.engine import (
    DEFAULT_CROP_H,
    DEFAULT_CROP_W,
    RPMLightEngine,
    build_52_panel_coords,
    build_depth_glow_boxes,
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


def test_52_panel_coords_count() -> None:
    coords = build_52_panel_coords()
    assert len(coords) == 52
    assert coords[0] == (0, 400, 0)


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
    assert r > g > 0
    assert b > 0


def test_falling_baseline_and_equalized_black_screen() -> None:
    engine = RPMLightEngine()
    # Start with a bright vivid red frame
    bright_red = _make_i420_frame(y_val=110, u_val=100, v_val=200)
    res1 = engine.process_i420_frame(bright_red)
    assert len(res1.zones_rgb) == 52
    assert res1.zones_rgb[0][0] > res1.zones_rgb[0][2]

    # Cut to dark screen: all zones first equalize toward the ~36 baseline and then slowly decay together
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
