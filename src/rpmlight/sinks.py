"""Output Sinks for RPMLight (Nanoleaf 4D UDP v2, WLED Realtime UDP, MiniMQTT, and Kodi OSD Toast)."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
import urllib.request


def _mqtt_encode_len(length: int) -> bytes:
    out = bytearray()
    while True:
        digit = length % 128
        length //= 128
        if length > 0:
            digit |= 0x80
        out.append(digit)
        if length == 0:
            break
    return bytes(out)


def _mqtt_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(">H", len(b)) + b


class MiniMqttPublisher:
    """Thread-safe zero-dependency raw socket MQTT v3.1.1 publisher for ceiling Sky Hue lights & HA status."""

    def __init__(
        self,
        host: str,
        port: int = 1883,
        user: str = "",
        password: str = "",
        client_id: str = "rpmlight-diamond-52",
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._client_id = client_id
        self._sock: socket.socket | None = None
        self._last_ping = 0.0
        self._lock = threading.RLock()

    def _connect(self) -> bool:
        with self._lock:
            try:
                self.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.5)
                s.connect((self._host, self._port))
                flags = 0xC2 if self._user else 0x02
                var_hdr = b"\x00\x04MQTT\x04" + bytes([flags]) + b"\x00\x3c"
                payload = _mqtt_str(self._client_id)
                if self._user:
                    payload += _mqtt_str(self._user) + _mqtt_str(self._password)
                pkt = (
                    b"\x10"
                    + _mqtt_encode_len(len(var_hdr) + len(payload))
                    + var_hdr
                    + payload
                )
                s.sendall(pkt)
                resp = s.recv(4)
                if len(resp) == 4 and resp[0] == 0x20 and resp[3] == 0x00:
                    s.settimeout(0.2)
                    self._sock = s
                    self._last_ping = time.monotonic()
                    return True
            except Exception:
                pass
            self.close()
            return False

    def publish(self, topic: str, message: str, retain: bool = False) -> None:
        with self._lock:
            if self._sock is None and not self._connect():
                return
            sock = self._sock
            if sock is None:
                return
            try:
                t_bytes = _mqtt_str(topic)
                m_bytes = message.encode("utf-8")
                body = t_bytes + m_bytes
                hdr = b"\x31" if retain else b"\x30"
                pkt = hdr + _mqtt_encode_len(len(body)) + body
                sock.sendall(pkt)
                now = time.monotonic()
                if now - self._last_ping >= 25.0:
                    sock.sendall(b"\xc0\x00")
                    self._last_ping = now
                    try:
                        sock.recv(16)
                    except Exception:
                        pass
            except Exception:
                self.close()

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


class NanoleafUdpSink:
    """Nanoleaf 4D / Shapes / Lines External Control v2 UDP Streamer (Port 60222)."""

    def __init__(
        self,
        ip: str,
        token: str = "",
        http_port: int = 16021,
        udp_port: int = 60222,
    ) -> None:
        self.ip = ip
        self.token = token
        self.http_port = http_port
        self.udp_port = udp_port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (ip, udp_port)
        self._http_lock = threading.Lock()
        self._http_event = threading.Event()
        self._pending_http_op: tuple[str, int] | None = None
        self._http_thread: threading.Thread | None = None

    def _ensure_http_worker(self) -> None:
        if not self.token:
            return
        with self._http_lock:
            if self._http_thread is not None and self._http_thread.is_alive():
                return
            self._http_thread = threading.Thread(
                target=self._http_worker_loop,
                name="RPMLight-NanoleafHTTP",
                daemon=True,
            )
            self._http_thread.start()

    def _http_worker_loop(self) -> None:
        while True:
            self._http_event.wait(timeout=5.0)
            self._http_event.clear()
            with self._http_lock:
                op = self._pending_http_op
                self._pending_http_op = None
            if op is None:
                continue
            kind, bri = op
            if kind == "activate":
                self.activate_extcontrol(brightness=bri)
            elif kind == "brightness":
                self.set_brightness(bri)
            elif kind == "off":
                self.turn_off()

    def _enqueue_http_op(self, kind: str, brightness: int = 85) -> None:
        if not self.token:
            return
        self._ensure_http_worker()
        with self._http_lock:
            # Coalesce consecutive background ops, preserving 'activate' over bare 'brightness'
            if (
                self._pending_http_op is not None
                and self._pending_http_op[0] == "activate"
                and kind == "brightness"
            ):
                kind = "activate"
            self._pending_http_op = (kind, int(brightness))
        self._http_event.set()

    def activate_extcontrol_async(self, brightness: int = 85) -> None:
        """Schedule External Control v2 activation off the real-time frame loop."""
        self._enqueue_http_op("activate", brightness)

    def set_brightness_async(self, brightness: int) -> None:
        """Schedule Master brightness update off the real-time frame/bus thread."""
        self._enqueue_http_op("brightness", brightness)

    def turn_off_async(self) -> None:
        """Schedule Nanoleaf power-off asynchronously."""
        self._enqueue_http_op("off", 0)

    def activate_extcontrol(self, brightness: int = 85) -> bool:
        """Enable External Control v2 via the Nanoleaf HTTP API."""
        if not self.token:
            return False
        base_url = f"http://{self.ip}:{self.http_port}/api/v1/{self.token}"
        try:
            bri = max(1, min(100, int(brightness)))
            state_payload = json.dumps(
                {"on": {"value": True}, "brightness": {"value": bri}}
            ).encode("utf-8")
            req_state = urllib.request.Request(
                f"{base_url}/state",
                data=state_payload,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req_state, timeout=4).close()

            eff_payload = json.dumps(
                {
                    "write": {
                        "command": "display",
                        "animType": "extControl",
                        "extControlVersion": "v2",
                    }
                }
            ).encode("utf-8")
            req_eff = urllib.request.Request(
                f"{base_url}/effects",
                data=eff_payload,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req_eff, timeout=4).close()
            return True
        except Exception:
            return False

    def set_brightness(self, brightness: int) -> bool:
        """Set Master hardware brightness (1..100%) on the Nanoleaf controller."""
        if not self.token:
            return False
        base_url = f"http://{self.ip}:{self.http_port}/api/v1/{self.token}"
        try:
            bri = max(1, min(100, int(brightness)))
            payload = json.dumps({"brightness": {"value": bri}}).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/state",
                data=payload,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=3).close()
            return True
        except Exception:
            return False

    def turn_off(self) -> bool:
        """Turn off the Nanoleaf controller cleanly."""
        if not self.token:
            return False
        base_url = f"http://{self.ip}:{self.http_port}/api/v1/{self.token}"
        try:
            payload = json.dumps({"on": {"value": False}}).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/state",
                data=payload,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=3).close()
            return True
        except Exception:
            return False

    @staticmethod
    def encode_v2_packet(
        zones_rgb: list[tuple[int, int, int]], transition_ds: int = 1
    ) -> bytes:
        """Encode a Nanoleaf External Control v2 UDP packet."""
        parts = [struct.pack(">H", len(zones_rgb))]
        for i, (r, g, b) in enumerate(zones_rgb):
            parts.append(struct.pack(">HBBBBH", i, r, g, b, 0, transition_ds))
        return b"".join(parts)

    def send_zones(
        self, zones_rgb: list[tuple[int, int, int]], transition_ds: int = 1
    ) -> None:
        pkt = self.encode_v2_packet(zones_rgb, transition_ds=transition_ds)
        self._sock.sendto(pkt, self._target)


class WledUdpSink:
    """WLED Realtime UDP DRGB Streamer (Port 21324)."""

    def __init__(self, ip: str, port: int = 21324, timeout_s: int = 2) -> None:
        self.ip = ip
        self.port = port
        self.timeout_s = max(1, min(255, timeout_s))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (ip, port)

    def encode_drgb_packet(self, zones_rgb: list[tuple[int, int, int]]) -> bytes:
        """Encode a WLED DRGB (protocol 2) UDP packet."""
        hdr = bytearray([2, self.timeout_s])
        for r, g, b in zones_rgb:
            hdr.extend((r & 0xFF, g & 0xFF, b & 0xFF))
        return bytes(hdr)

    def send_zones(self, zones_rgb: list[tuple[int, int, int]]) -> None:
        self._sock.sendto(self.encode_drgb_packet(zones_rgb), self._target)


def _normalize_kodi_endpoint(
    host: str, default_port: int = 8080
) -> tuple[str, int, str]:
    """Normalize ``host``, ``host:port``, or full ``http://host:port/jsonrpc`` into ``(host, port, url)``."""
    raw = host.strip()
    scheme = "http"
    if raw.startswith("https://"):
        scheme = "https"
        raw = raw[len("https://") :]
    elif raw.startswith("http://"):
        raw = raw[len("http://") :]
    raw = raw.split("/", 1)[0].strip()
    parsed_port = int(default_port)
    parsed_host = raw
    if ":" in raw and not raw.endswith("]"):
        maybe_host, maybe_port = raw.rsplit(":", 1)
        if maybe_port.isdigit():
            parsed_host = maybe_host
            parsed_port = int(maybe_port)
    return parsed_host, parsed_port, f"{scheme}://{parsed_host}:{parsed_port}/jsonrpc"


class KodiJsonRpcToastSink:
    """Direct local Kodi JSON-RPC OSD notification sink (never switches TV HDMI input)."""

    def __init__(
        self, host: str, port: int = 8080, display_time_ms: int = 4500
    ) -> None:
        self.host, self.port, self.url = _normalize_kodi_endpoint(host, port)
        self.display_time_ms = display_time_ms

    def send_toast(self, title: str, message: str, image: str = "info") -> bool:
        """Display an on-screen toast overlay on Kodi via GUI.ShowNotification."""
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "GUI.ShowNotification",
                "params": {
                    "title": title,
                    "message": message,
                    "image": image,
                    "displaytime": self.display_time_ms,
                },
            }
        ).encode("utf-8")
        try:
            req = urllib.request.Request(
                self.url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=1.5).close()
            return True
        except Exception:
            return False
