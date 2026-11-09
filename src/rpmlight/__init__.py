"""RPMLight: Perceptual Zero-Blink Camera & Screen Ambilight Engine."""

from rpmlight.engine import (
    RPMLightEngine,
    SliderCurveParams,
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
)
from rpmlight.sinks import (
    KodiJsonRpcToastSink,
    MiniMqttPublisher,
    NanoleafUdpSink,
    WledUdpSink,
)

__version__ = "0.3.0"

__all__ = [
    "KodiJsonRpcToastSink",
    "MiniMqttPublisher",
    "NanoleafUdpSink",
    "RPMLightEngine",
    "SliderCurveParams",
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
    "hue_color_and_bri",
    "sample_box_fast",
]
