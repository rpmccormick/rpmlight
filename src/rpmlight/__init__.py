"""RPMLight: Perceptual Zero-Blink Camera & Screen Ambilight Engine."""

from rpmlight.atmosphere import (
    CastV2PhotoSink,
    delay_s_to_slider_pct,
    is_allowed_personal_album_url,
    parse_google_photos_album_html,
    slider_pct_to_delay_s,
)
from rpmlight.brilliant_plugin import (
    BrilliantAmbiPluginState,
    start_brilliant_virtual_light,
)
from rpmlight.engine import (
    RPMLightEngine,
    SliderCurveParams,
    SubBassBeatDetector,
    ZoneFrameResult,
    apply_slider_curve,
    build_52_panel_coords,
    build_auto_crop_depth_boxes,
    build_depth_glow_boxes,
    build_diamond_boxes,
    build_sampling_boxes,
    describe_slider_setting,
    hue_color_and_bri,
    sample_box_fast,
    suppress_non_dominant,
)
from rpmlight.sinks import (
    KodiJsonRpcToastSink,
    MiniMqttPublisher,
    NanoleafUdpSink,
    WledUdpSink,
)
from rpmlight.tpu import (
    CropCalibrator,
    CropMargins,
    TPUaaSBackgroundWorker,
    TPUSceneCue,
    estimate_i420_active_bbox_norm,
    i420_to_ppm,
    infer_scene_cue,
    query_gemini_vision_cue,
)

__version__ = "0.6.0"

__all__ = [
    "BrilliantAmbiPluginState",
    "CastV2PhotoSink",
    "CropCalibrator",
    "CropMargins",
    "KodiJsonRpcToastSink",
    "MiniMqttPublisher",
    "NanoleafUdpSink",
    "RPMLightEngine",
    "SliderCurveParams",
    "SubBassBeatDetector",
    "TPUSceneCue",
    "TPUaaSBackgroundWorker",
    "WledUdpSink",
    "ZoneFrameResult",
    "__version__",
    "apply_slider_curve",
    "build_52_panel_coords",
    "build_auto_crop_depth_boxes",
    "build_depth_glow_boxes",
    "build_diamond_boxes",
    "build_sampling_boxes",
    "describe_slider_setting",
    "estimate_i420_active_bbox_norm",
    "hue_color_and_bri",
    "i420_to_ppm",
    "infer_scene_cue",
    "query_gemini_vision_cue",
    "sample_box_fast",
    "start_brilliant_virtual_light",
    "suppress_non_dominant",
]
