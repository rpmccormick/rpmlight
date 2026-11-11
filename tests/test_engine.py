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
from rpmlight.tpu import (
    CropCalibrator,
    CropMargins,
    TPUaaSBackgroundWorker,
    estimate_i420_active_bbox_norm,
    i420_to_ppm,
    infer_scene_cue,
)


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
        mv[:y_size],
        mv[y_size : y_size + uv_size],
        mv[y_size + uv_size :],
        ey_idx,
        euv_idx,
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
        mv_b[:y_size],
        mv_b[y_size : y_size + uv_size],
        mv_b[y_size + uv_size :],
        ey_idx,
        euv_idx,
    )
    assert g_b >= int(r_b * 0.55)
    assert b_b > 0

    # True 650nm OLED Red (Terminator POV vision: y=15, u=125 [-3], v=136 [+8]) must register as vivid crimson red
    frame_t3 = _make_i420_frame(y_val=15, u_val=125, v_val=136)
    mv_t3 = memoryview(frame_t3)
    r_t3, g_t3, _b_t3 = sample_box_fast(
        mv_t3[:y_size],
        mv_t3[y_size : y_size + uv_size],
        mv_t3[y_size + uv_size :],
        ey_idx,
        euv_idx,
    )
    assert r_t3 >= 100
    assert r_t3 > g_t3 * 2.5

    # Symmetric +v / -v Bayer demosaicing fringe at equal luma cancels to neutral (no red bias)
    frame_fringe = _make_i420_frame(y_val=28, u_val=128, v_val=128)
    for idx, uv_i in enumerate(euv_idx):
        frame_fringe[y_size + uv_size + uv_i] = 148 if (idx % 2 == 0) else 108
    mv_f = memoryview(frame_fringe)
    r_f, g_f, _b_f = sample_box_fast(
        mv_f[:y_size],
        mv_f[y_size : y_size + uv_size],
        mv_f[y_size + uv_size :],
        ey_idx,
        euv_idx,
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


def test_tpuaas_i420_to_ppm_and_scene_cue_inference() -> None:
    frame = _make_i420_frame(y_val=120, u_val=128, v_val=128)
    ppm = i420_to_ppm(frame, step=2)
    assert ppm.startswith(b"P6\n160 88\n255\n")
    assert len(ppm) == len(b"P6\n160 88\n255\n") + 160 * 88 * 3

    # 1. Close-up dialogue (1 large person bbox >= 18% of screen) -> "drama"
    cue_drama = infer_scene_cue(
        {
            "detections": [
                {
                    "label": "person",
                    "score": 0.88,
                    "bbox_norm": [0.25, 0.15, 0.75, 0.85],
                }
            ],
            "tpu_ms": 5.94,
            "sram_cache_hit": True,
        }
    )
    assert cue_drama.scene_class == "drama"
    assert cue_drama.sram_cache_hit is True
    assert abs(cue_drama.focal_x - 0.5) < 0.05

    # 2. Sports / fast action on left side -> "music_video" + left focal_x bias
    cue_action = infer_scene_cue(
        {
            "detections": [
                {
                    "label": "person",
                    "score": 0.82,
                    "bbox_norm": [0.05, 0.30, 0.25, 0.70],
                },
                {
                    "label": "sports ball",
                    "score": 0.75,
                    "bbox_norm": [0.10, 0.40, 0.18, 0.52],
                },
            ],
            "tpu_ms": 5.94,
            "sram_cache_hit": True,
        }
    )
    assert cue_action.scene_class == "music_video"
    assert cue_action.focal_x < 0.30
    assert cue_action.wing_weights["west"] > cue_action.wing_weights["east"]

    # 3. Engine applies TPU cue (scene_class + wing_weights) and emits lightshow diamond dict
    engine = RPMLightEngine(slider_pct=50)
    engine.apply_tpu_cue(cue_action)
    assert engine.scene_class == "music_video"
    assert engine.wing_weights["west"] > engine.wing_weights["east"]
    res = engine.process_i420_frame(frame)
    diamond = res.to_diamond_dict(slider_pct=50)
    assert diamond["scene_class"] == "music_video"
    assert "beat_pulse" in diamond and "energy" in diamond

    # 4. Optional Auto-Crop preserves hardcoded crop and reverts cleanly
    hardcoded = CropMargins(top=328, bottom=216, left=608, right=352)
    cal = CropCalibrator(hardcoded, auto_crop_enabled=False)
    assert cal.current_crop == hardcoded
    auto_applied = cal.apply_auto_crop_bbox((0.25, 0.25, 0.75, 0.75))
    assert auto_applied != hardcoded
    assert cal.saved_crop == hardcoded
    reverted = cal.revert_crop()
    assert reverted == hardcoded

    # 5. Offline pure-Python I420 active content bbox estimation
    letterbox_frame = _make_i420_frame(y_val=4, u_val=128, v_val=128)
    _paint_i420_rect(letterbox_frame, 32, 28, 288, 148, y_val=120, u_val=128, v_val=128)
    bbox = estimate_i420_active_bbox_norm(letterbox_frame)
    assert 0.08 <= bbox[0] <= 0.15
    assert 0.12 <= bbox[1] <= 0.22
    assert 0.85 <= bbox[2] <= 0.94
    assert 0.80 <= bbox[3] <= 0.90

    # 6. TPUaaSBackgroundWorker is disabled by default when no URL is configured
    worker = TPUaaSBackgroundWorker(base_url="")
    assert worker.enabled is False


def test_pause_freeze_gate_and_sub_bass_beat_detector() -> None:
    import math
    import struct
    from rpmlight.engine import SubBassBeatDetector, suppress_non_dominant

    # 1. Dominant-channel power-curve suppression scales cleanly from 0.0 (50% natural) to 1.0 (100% neon/red)
    r0, g0, b0 = suppress_non_dominant(250.0, 120.0, 80.0, strength=0.0)
    assert (r0, g0, b0) == (250.0, 120.0, 80.0)
    r1, g1, b1 = suppress_non_dominant(250.0, 120.0, 80.0, strength=1.0)
    assert r1 == 250.0 and g1 < 35.0 and b1 < 15.0

    # 2. Pause freeze gate locks after 10 steady frames and unlocks on >4% change
    engine = RPMLightEngine(slider_pct=50)
    frame_a = _make_i420_frame(y_val=100, u_val=100, v_val=200)
    for _ in range(15):
        res = engine.process_i420_frame(frame_a)
    assert res.is_video_paused is True
    locked_rgb = list(res.zones_rgb)
    res_still = engine.process_i420_frame(frame_a)
    assert res_still.zones_rgb == locked_rgb

    frame_b = _make_i420_frame(y_val=120, u_val=210, v_val=90)
    res_play = engine.process_i420_frame(frame_b)
    assert res_play.is_video_paused is False

    # 3. SubBassBeatDetector detects a 65 Hz sub-bass kick
    det = SubBassBeatDetector()
    quiet_pcm = struct.pack("<800h", *([10] * 800))
    for _ in range(5):
        det.process_pcm_s16le(quiet_pcm)
    kick_samples = [
        int(4500 * math.sin(2.0 * math.pi * 65.0 * i / 16000.0)) for i in range(800)
    ]
    kick_pcm = struct.pack("<800h", *kick_samples)
    pulse, strength, energy = det.process_pcm_s16le(kick_pcm)
    assert pulse is True
    assert strength >= 0.35
    assert energy > 0.0


def test_cli_and_brilliant_plugin_modes() -> None:
    from rpmlight.brilliant_plugin import (
        BrilliantAmbiPluginState,
        start_brilliant_virtual_light,
    )
    from rpmlight.cli import build_parser

    # 1. Standalone Linux V4L2 mode
    args_v4l2 = build_parser().parse_args(
        ["--source", "v4l2", "--device", "/dev/video2", "--slider-pct", "75"]
    )
    assert args_v4l2.source == "v4l2"
    assert args_v4l2.device == "/dev/video2"
    assert args_v4l2.slider_pct == 75
    assert args_v4l2.brilliant_plugin is False

    # 2. Brilliant Panel Plugin mode
    args_panel = build_parser().parse_args(
        ["--brilliant-plugin", "--start-off", "--room-id", "room-123"]
    )
    assert args_panel.brilliant_plugin is True
    assert args_panel.start_off is True
    assert args_panel.room_id == "room-123"

    # 3. Graceful fallback on non-Brilliant machines
    st = BrilliantAmbiPluginState(room_id="room-123", enabled=False, slider_pct=50)
    assert start_brilliant_virtual_light(st) is False
