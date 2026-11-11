"""Command-line interface and daemon runner for RPMLight (Standalone Linux & Brilliant Panel Plugin)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from rpmlight.brilliant_plugin import (
    BrilliantAmbiPluginState,
    start_brilliant_virtual_light,
)
from rpmlight.engine import DEFAULT_CROP_H, DEFAULT_CROP_W, RPMLightEngine
from rpmlight.sinks import (
    KodiJsonRpcToastSink,
    MiniMqttPublisher,
    NanoleafUdpSink,
    WledUdpSink,
)
from rpmlight.sources import (
    build_brilliant_imx_gst_cmd,
    build_ffmpeg_i420_cmd,
    build_v4l2_gst_cmd,
)
from rpmlight.tpu import TPUaaSBackgroundWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rpmlight",
        description=(
            "RPMLight: Perceptual Zero-Blink 52-Zone Camera & Screen Ambilight Engine "
            "(Standalone Linux V4L2/FFmpeg Service & Brilliant Panel Plugin)"
        ),
    )
    parser.add_argument(
        "--source",
        choices=("brilliant", "v4l2", "ffmpeg"),
        default=os.environ.get("RPMLIGHT_SOURCE", "brilliant"),
        help="Video capture pipeline backend: brilliant (i.MX6 ISP), v4l2 (USB/Linux camera), or ffmpeg (RTSP/HTTP)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("RPMLIGHT_DEVICE", "/dev/video0"),
        help="Camera device path or RTSP/stream URL (default: /dev/video0)",
    )
    parser.add_argument(
        "--brilliant-plugin",
        action="store_true",
        default=os.environ.get("RPMLIGHT_BRILLIANT_PLUGIN", "").lower()
        in ("1", "true", "yes"),
        help="Host native 'TV AmbiLED' virtual dimmer light on the Brilliant panel Thrift bus",
    )
    parser.add_argument(
        "--room-id",
        default=os.environ.get("RPMLIGHT_ROOM_ID", ""),
        help="Optional Brilliant room_id for the hosted 'TV AmbiLED' virtual light",
    )
    parser.add_argument(
        "--start-off",
        action="store_true",
        default=os.environ.get("RPMLIGHT_START_OFF", "").lower()
        in ("1", "true", "yes"),
        help="Start in Camera Sleep Mode (/dev/video0 closed) until 'TV AmbiLED' is turned ON",
    )
    parser.add_argument(
        "--crop-x", type=int, default=int(os.environ.get("RPMLIGHT_CROP_X", "584"))
    )
    parser.add_argument(
        "--crop-y", type=int, default=int(os.environ.get("RPMLIGHT_CROP_Y", "272"))
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("RPMLIGHT_WIDTH", str(DEFAULT_CROP_W))),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("RPMLIGHT_HEIGHT", str(DEFAULT_CROP_H))),
    )
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
        help="Optional MQTT broker host for 4-Zone Hue Sky/Wing sync and status telemetry",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=int(os.environ.get("RPMLIGHT_MQTT_PORT", "1883")),
    )
    parser.add_argument("--mqtt-user", default=os.environ.get("RPMLIGHT_MQTT_USER", ""))
    parser.add_argument("--mqtt-pass", default=os.environ.get("RPMLIGHT_MQTT_PASS", ""))
    parser.add_argument(
        "--mqtt-topic",
        default=os.environ.get(
            "RPMLIGHT_MQTT_TOPIC", "rpmlight/diamond_ambilight/state"
        ),
    )
    parser.add_argument(
        "--tpu-url",
        default=os.environ.get("RPMLIGHT_TPU_URL", ""),
        help="Optional Coral EdgeTPU (TPUaaS) URL for low-rate 0.25Hz scene cues",
    )
    parser.add_argument(
        "--kodi-host",
        default=os.environ.get("RPMLIGHT_KODI_HOST", ""),
        help="Optional Kodi JSON-RPC host:port for on-screen slider toast notifications",
    )
    parser.add_argument(
        "--slider-pct",
        type=int,
        default=int(os.environ.get("RPMLIGHT_SLIDER_PCT", "50")),
        help="Perceptual slider setting 1..100 (default: 50 Balanced Sweet Spot)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    width = args.width
    height = args.height
    frame_bytes = (width * height * 3) // 2
    gst_buffer_bytes = frame_bytes + 9600 if args.source == "brilliant" else frame_bytes

    engine = RPMLightEngine(width=width, height=height, slider_pct=args.slider_pct)
    nl_sink = (
        NanoleafUdpSink(args.nanoleaf_ip, args.nanoleaf_token)
        if args.nanoleaf_ip
        else None
    )
    wled_sink = WledUdpSink(args.wled_ip) if args.wled_ip else None
    mqtt_pub = (
        MiniMqttPublisher(
            args.mqtt_host, args.mqtt_port, args.mqtt_user, args.mqtt_pass
        )
        if args.mqtt_host
        else None
    )
    kodi_toast = KodiJsonRpcToastSink(args.kodi_host) if args.kodi_host else None
    tpu_worker = TPUaaSBackgroundWorker(
        base_url=args.tpu_url, width=width, height=height
    )
    if args.tpu_url:
        tpu_worker.start()

    plugin_state: BrilliantAmbiPluginState | None = None
    if args.brilliant_plugin:

        def _on_plugin_change(_enabled: bool, new_pct: int) -> None:
            engine.set_slider_pct(new_pct)

        plugin_state = BrilliantAmbiPluginState(
            room_id=args.room_id,
            enabled=not args.start_off,
            slider_pct=args.slider_pct,
            mqtt_pub=mqtt_pub,
            kodi_toast=kodi_toast,
            on_change=_on_plugin_change,
        )
        start_brilliant_virtual_light(plugin_state)

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
        if plugin_state is not None and not plugin_state.enabled:
            print(
                "Camera Sleep Mode active (/dev/video0 closed, red LED OFF). "
                "Waiting for 'TV AmbiLED' ON...",
                flush=True,
            )
            while not plugin_state.enabled:
                time.sleep(0.25)

        if nl_sink is not None:
            nl_sink.activate_extcontrol(brightness=engine.nl_Master_brightness)
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
            import fcntl

            fcntl.fcntl(fd, 1031, 4096)
        except Exception:
            pass

        buf = bytearray(gst_buffer_bytes)
        mv = memoryview(buf)
        frames = 0
        t0 = time.monotonic()
        try:
            while True:
                if plugin_state is not None and not plugin_state.enabled:
                    if nl_sink is not None:
                        nl_sink.send_zones([(0, 0, 0)] * 52, transition_ds=1)
                    if mqtt_pub is not None:
                        mqtt_pub.publish(
                            args.mqtt_topic,
                            json.dumps(
                                {
                                    "restore": True,
                                    "lightshow_active": False,
                                    "beat_pulse": False,
                                    "energy": 0.0,
                                }
                            ),
                        )
                    break

                pos = 0
                while pos < gst_buffer_bytes:
                    chunk = os.read(fd, gst_buffer_bytes - pos)
                    if not chunk:
                        raise EOFError("Video capture pipe closed")
                    buf[pos : pos + len(chunk)] = chunk
                    pos += len(chunk)

                frames += 1
                frame_slice = mv[0:frame_bytes]
                if args.tpu_url:
                    tpu_worker.offer_i420_frame(bytes(frame_slice))
                    if tpu_worker.latest_cue is not None:
                        engine.apply_tpu_cue(tpu_worker.latest_cue)

                res = engine.process_i420_frame(frame_slice)
                if nl_sink is not None:
                    nl_sink.send_zones(
                        res.zones_rgb, transition_ds=res.nl_transition_ds
                    )
                if wled_sink is not None:
                    wled_sink.send_zones(res.zones_rgb)

                now = time.monotonic()
                b_norm = engine.blinkiness / 255.0
                hue_interval = max(0.30, 0.85 - 0.55 * b_norm)
                if mqtt_pub is not None and (
                    (res.is_scene_cut and now - last_hue_pub >= 0.30)
                    or (res.beat_pulse and now - last_hue_pub >= 0.18)
                    or (now - last_hue_pub >= hue_interval)
                ):
                    mqtt_pub.publish(
                        args.mqtt_topic,
                        json.dumps(res.to_diamond_dict()),
                    )
                    last_hue_pub = now

                if now - t0 >= 8.0:
                    fps = frames / (now - t0)
                    print(
                        f"RPMLight {len(res.zones_rgb)} zones @ {fps:.1f} FPS "
                        f"({engine.mode}, slider={engine.slider_pct}%) "
                        f"| bottom={res.zones_rgb[0]} top={res.zones_rgb[25]} | Sky={res.up_rgb}",
                        flush=True,
                    )
                    frames = 0
                    t0 = now

                if nl_sink is not None and now - last_ext_reassert >= 60.0:
                    nl_sink.activate_extcontrol(brightness=engine.nl_Master_brightness)
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


def main_brilliant_plugin() -> int:
    """Entry point for running RPMLight as a Brilliant Panel plugin service."""
    import sys

    argv = list(sys.argv[1:])
    if "--brilliant-plugin" not in argv:
        argv.insert(0, "--brilliant-plugin")
    if "--start-off" not in argv:
        argv.insert(1, "--start-off")
    return main(argv)
