"""Command-line interface and daemon runner for RPMLight."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time

from rpmlight.engine import DEFAULT_CROP_H, DEFAULT_CROP_W, RPMLightEngine
from rpmlight.sinks import MiniMqttPublisher, NanoleafUdpSink, WledUdpSink
from rpmlight.sources import (
    build_brilliant_imx_gst_cmd,
    build_ffmpeg_i420_cmd,
    build_v4l2_gst_cmd,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rpmlight",
        description="RPMLight: Perceptual Zero-Blink Camera & Screen Ambilight Streamer",
    )
    parser.add_argument(
        "--source",
        choices=("brilliant", "v4l2", "ffmpeg"),
        default=os.environ.get("RPMLIGHT_SOURCE", "brilliant"),
        help="Video capture pipeline backend (default: brilliant)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("RPMLIGHT_DEVICE", "/dev/video0"),
        help="Camera device path or RTSP/stream URL (default: /dev/video0)",
    )
    parser.add_argument("--crop-x", type=int, default=int(os.environ.get("RPMLIGHT_CROP_X", "584")))
    parser.add_argument("--crop-y", type=int, default=int(os.environ.get("RPMLIGHT_CROP_Y", "272")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("RPMLIGHT_WIDTH", str(DEFAULT_CROP_W))))
    parser.add_argument("--height", type=int, default=int(os.environ.get("RPMLIGHT_HEIGHT", str(DEFAULT_CROP_H))))
    parser.add_argument(
        "--nanoleaf-ip",
        default=os.environ.get("RPMLIGHT_NANOLEAF_IP", ""),
        help="Nanoleaf 4D / Shapes IP address for External Control v2 UDP (port 60222)",
    )
    parser.add_argument(
        "--nanoleaf-token",
        default=os.environ.get("RPMLIGHT_NANOLEAF_TOKEN", ""),
        help="Nanoleaf HTTP API auth token for automatic External Control v2 activation",
    )
    parser.add_argument(
        "--wled-ip",
        default=os.environ.get("RPMLIGHT_WLED_IP", ""),
        help="Optional WLED controller IP address for DRGB UDP streaming (port 21324)",
    )
    parser.add_argument(
        "--mqtt-host",
        default=os.environ.get("RPMLIGHT_MQTT_HOST", ""),
        help="Optional MQTT broker host for ceiling Sky Hue light synchronization",
    )
    parser.add_argument("--mqtt-port", type=int, default=int(os.environ.get("RPMLIGHT_MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-user", default=os.environ.get("RPMLIGHT_MQTT_USER", ""))
    parser.add_argument("--mqtt-pass", default=os.environ.get("RPMLIGHT_MQTT_PASS", ""))
    parser.add_argument(
        "--mqtt-topic",
        default=os.environ.get("RPMLIGHT_MQTT_TOPIC", "rpmlight/diamond_ambilight/state"),
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "super_chill", "extra_color", "true_edge"),
        default=os.environ.get("RPMLIGHT_MODE", "normal"),
    )
    parser.add_argument(
        "--blinkiness",
        type=int,
        default=int(os.environ.get("RPMLIGHT_BLINKINESS", "128")),
        help="Responsiveness / edge-accuracy scale 0..255 (default: 128)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    width = args.width
    height = args.height
    frame_bytes = (width * height * 3) // 2
    gst_buffer_bytes = frame_bytes + 9600 if args.source == "brilliant" else frame_bytes

    engine = RPMLightEngine(width=width, height=height, mode=args.mode, blinkiness=args.blinkiness)
    nl_sink = NanoleafUdpSink(args.nanoleaf_ip, args.nanoleaf_token) if args.nanoleaf_ip else None
    wled_sink = WledUdpSink(args.wled_ip) if args.wled_ip else None
    mqtt_pub = (
        MiniMqttPublisher(args.mqtt_host, args.mqtt_port, args.mqtt_user, args.mqtt_pass)
        if args.mqtt_host
        else None
    )

    if args.source == "brilliant":
        cmd = build_brilliant_imx_gst_cmd(
            device=args.device,
            crop_x=args.crop_x,
            crop_y=args.crop_y,
            crop_w=width,
            crop_h=height,
        )
    elif args.source == "v4l2":
        cmd = build_v4l2_gst_cmd(device=args.device, width=width, height=height)
    else:
        cmd = build_ffmpeg_i420_cmd(input_url=args.device, width=width, height=height)

    while True:
        if nl_sink is not None:
            nl_sink.activate_extcontrol()
        last_ext_reassert = time.monotonic()
        last_hue_pub = 0.0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=gst_buffer_bytes,
        )
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        try:
            fcntl.fcntl(fd, 1031, 4096)
        except Exception:
            pass

        buf = bytearray(gst_buffer_bytes)
        mv = memoryview(buf)
        frames = 0
        t0 = time.monotonic()
        try:
            while True:
                pos = 0
                while pos < gst_buffer_bytes:
                    chunk = os.read(fd, gst_buffer_bytes - pos)
                    if not chunk:
                        raise EOFError("Video capture pipe closed")
                    buf[pos : pos + len(chunk)] = chunk
                    pos += len(chunk)

                frames += 1
                res = engine.process_i420_frame(mv[0:frame_bytes])
                if nl_sink is not None:
                    nl_sink.send_zones(res.zones_rgb, transition_ds=res.nl_transition_ds)
                if wled_sink is not None:
                    wled_sink.send_zones(res.zones_rgb)

                now = time.monotonic()
                b_norm = engine.blinkiness / 255.0
                hue_interval = max(0.35, 1.60 - 1.20 * b_norm)
                hue_transition = 0.25 if res.is_scene_cut else max(0.35, 2.00 - 1.60 * b_norm)
                if mqtt_pub is not None and (
                    (res.is_scene_cut and now - last_hue_pub >= 0.30)
                    or (now - last_hue_pub >= hue_interval)
                ):
                    mqtt_pub.publish(
                        args.mqtt_topic,
                        json.dumps(
                            {
                                "up_rgb": res.up_rgb,
                                "up_bri": res.up_bri,
                                "down_rgb": res.down_rgb,
                                "down_bri": res.down_bri,
                                "transition": hue_transition,
                            }
                        ),
                    )
                    last_hue_pub = now

                if now - t0 >= 8.0:
                    fps = frames / (now - t0)
                    print(
                        f"RPMLight {len(res.zones_rgb)} zones @ {fps:.1f} FPS ({engine.mode}) "
                        f"| bottom={res.zones_rgb[0]} top={res.zones_rgb[25]} | Sky={res.up_rgb}",
                        flush=True,
                    )
                    frames = 0
                    t0 = now

                if nl_sink is not None and now - last_ext_reassert >= 60.0:
                    nl_sink.activate_extcontrol()
                    last_ext_reassert = now
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(f"Stream loop error: {exc}", flush=True)
        finally:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
            time.sleep(1.0)
