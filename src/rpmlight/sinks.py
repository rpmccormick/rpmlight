"""Output Sinks for RPMLight (Nanoleaf 4D UDP v2, WLED Realtime UDP, MiniMQTT, and Kodi OSD Toast)."""

from __future__ import annotations

import json
import socket
import struct
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
    """Zero-dependency raw socket MQTT v3.1.1 publisher for ceiling Sky Hue lights & HA status."""

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

    def _connect(self) -> bool:
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
            pkt = b"\x10" + _mqtt_encode_len(len(var_hdr) + len(payload)) + var_hdr + payload
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

    def publish(self, topic: str, message: str) -> None:
        if self._sock is None and not self._connect():
            return
        assert self._sock is not None
        try:
            t_bytes = _mqtt_str(topic)
            m_bytes = message.encode("utf-8")
            body = t_bytes + m_bytes
            pkt = b"\x30" + _mqtt_encode_len(len(body)) + body
            self._sock.sendall(pkt)
            now = time.monotonic()
            if now - self._last_ping >= 25.0:
                self._sock.sendall(b"\xc0\x00")
                self._last_ping = now
                try:
                    self._sock.recv(16)
                except Exception:
                    pass
        except Exception:
            self.close()

    def close(self) -> None:
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

    def activate_extcontrol(self, brightness: int = 85) -> bool:
        """Enable External Control v2 via the Nanoleaf HTTP API."""
        if not self.token:
            return False
        base_url = f"http://{self.ip}:{self.http_port}/api/v1/{self.token}"
        try:
            bri = max(1, min(100, int(brightness)))
            state_payload = json.dumps({"on": {"value": True}, "brightness": {"value": bri}}).encode(
                "utf-8"
            )
            req_state = urllib.request.Request(
                f"{base_url}/state",
                data=state_payload,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req_state, timeout=4).close()

            eff_payload = json.dumps(
                {"write": {"command": "display", "animType": "extControl", "extControlVersion": "v2"}}
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


class KodiJsonRpcToastSink:
    """Direct local Kodi JSON-RPC OSD notification sink (never switches TV HDMI input)."""

    def __init__(self, host: str, port: int = 8080, display_time_ms: int = 4500) -> None:
        self.host = host
        self.port = port
        self.display_time_ms = display_time_ms
        self.url = f"http://{host}:{port}/jsonrpc"

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
