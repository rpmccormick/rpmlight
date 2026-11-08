"""Video / Camera Frame Sources for RPMLight (Brilliant i.MX6 ISP, V4L2, RTSP/FFmpeg)."""

from __future__ import annotations


def build_brilliant_imx_gst_cmd(
    device: str = "/dev/video0",
    crop_x: int = 584,
    crop_y: int = 272,
    crop_w: int = 320,
    crop_h: int = 176,
    gain: float = 3.4,
    exposure: int = 68,
) -> list[str]:
    """Build the zero-latency GStreamer pipeline for Brilliant Control i.MX6 panels."""
    return [
        "gst-launch-1.0",
        "-q",
        "imxv4l2videosrc",
        f"device={device}",
        "!",
        "video/x-bayer,format=bggr,width=1280,height=720,framerate=30/1",
        "!",
        "bayer2yuv",
        "demosaicing=1",
        "eg-mode=1",
        f"gain={gain}",
        f"exposure={exposure}",
        "wb-mode=1",
        "ct-mode=1",
        f"crop-meta-x={crop_x}",
        f"crop-meta-y={crop_y}",
        f"crop-meta-width={crop_w}",
        f"crop-meta-height={crop_h}",
        "!",
        "queue",
        "max-size-buffers=1",
        "leaky=downstream",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


def build_v4l2_gst_cmd(
    device: str = "/dev/video0",
    width: int = 320,
    height: int = 176,
    fps: int = 30,
) -> list[str]:
    """Build a standard Linux V4L2 GStreamer pipeline outputting raw I420 frames."""
    return [
        "gst-launch-1.0",
        "-q",
        "v4l2src",
        f"device={device}",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,format=I420,width={width},height={height},framerate={fps}/1",
        "!",
        "queue",
        "max-size-buffers=1",
        "leaky=downstream",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


def build_ffmpeg_i420_cmd(
    input_url: str,
    width: int = 320,
    height: int = 176,
    fps: int = 20,
) -> list[str]:
    """Build an FFmpeg pipeline outputting raw I420 (yuv420p) frames to stdout."""
    return [
        "ffmpeg",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-i",
        input_url,
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
