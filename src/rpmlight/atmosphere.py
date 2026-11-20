"""Standalone On-Panel Atmosphere CastV2 Photo Sender, Delay Curve & Album Parser."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import math
import re
import socket
import ssl
import struct
import threading
import time
import urllib.parse

MIN_PHOTO_DELAY_S = 5.0
MAX_PHOTO_DELAY_S = 300.0
DEFAULT_SLIDER_PCT = 50
DEFAULT_AMBILED_BRIGHTNESS_PCT = 85

_SLIDER_CURVE_POINTS: tuple[tuple[float, float], ...] = (
    (1.0, MAX_PHOTO_DELAY_S),
    (25.0, 120.0),
    (50.0, 45.0),
    (75.0, 15.0),
    (100.0, MIN_PHOTO_DELAY_S),
)

_ALLOWED_ALBUM_HOSTS = frozenset(
    {"photos.app.goo.gl", "photos.google.com", "www.icloud.com"}
)
_ICLOUD_SHARED_HOST_RE = re.compile(r"^p\d+-sharedstreams\.icloud\.com$", re.IGNORECASE)
_GPHOTOS_TUPLE_RE = re.compile(
    r'\["(https://lh3\.googleusercontent\.com/(?:pw/)?[A-Za-z0-9_-]{20,})",\s*(\d+),\s*(\d+)'
)
_GPHOTOS_URL_RE = re.compile(
    r"(https://lh3\.googleusercontent\.com/pw/[A-Za-z0-9_-]{20,})"
)


def slider_pct_to_delay_s(pct: float) -> float:
    """Map Atmosphere brightness slider (``1..100%``) to photo delay (``300s..5s``)."""
    if not math.isfinite(pct):
        return 45.0
    clamped = max(1.0, min(100.0, float(pct)))
    for idx in range(len(_SLIDER_CURVE_POINTS) - 1):
        p0, d0 = _SLIDER_CURVE_POINTS[idx]
        p1, d1 = _SLIDER_CURVE_POINTS[idx + 1]
        if p0 <= clamped <= p1:
            return round(d0 + ((clamped - p0) / (p1 - p0)) * (d1 - d0), 1)
    return MIN_PHOTO_DELAY_S


def delay_s_to_slider_pct(delay_s: float) -> int:
    """Map photo delay (``5..300s``) back to slider percentage (``1..100%``)."""
    if not math.isfinite(delay_s):
        return DEFAULT_SLIDER_PCT
    clamped = max(MIN_PHOTO_DELAY_S, min(MAX_PHOTO_DELAY_S, float(delay_s)))
    for idx in range(len(_SLIDER_CURVE_POINTS) - 1):
        p0, d0 = _SLIDER_CURVE_POINTS[idx]
        p1, d1 = _SLIDER_CURVE_POINTS[idx + 1]
        if d1 <= clamped <= d0:
            return max(
                1, min(100, round(p0 + ((d0 - clamped) / (d0 - d1)) * (p1 - p0)))
            )
    return DEFAULT_SLIDER_PCT


def is_allowed_personal_album_url(url: str) -> bool:
    """Return ``True`` only for HTTPS Google Photos or iCloud shared album links."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    with contextlib.suppress(ValueError):
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return host in _ALLOWED_ALBUM_HOSTS or bool(_ICLOUD_SHARED_HOST_RE.match(host))


def parse_google_photos_album_html(html: str, *, max_photos: int = 100) -> list[str]:
    """Extract deduplicated ``=w1920-h1080`` URLs from a Google Photos shared album page."""
    landscape: list[str] = []
    other: list[str] = []
    seen: set[str] = set()
    for match in _GPHOTOS_TUPLE_RE.finditer(html):
        base_url = match.group(1)
        if base_url in seen:
            continue
        seen.add(base_url)
        width, height = int(match.group(2)), int(match.group(3))
        full = f"{base_url}=w1920-h1080"
        if width >= int(height * 0.8):
            landscape.append(full)
        else:
            other.append(full)
        if len(landscape) + len(other) >= max_photos:
            break
    if not seen:
        for match in _GPHOTOS_URL_RE.finditer(html):
            base_url = match.group(1)
            if base_url not in seen:
                seen.add(base_url)
                landscape.append(f"{base_url}=w1920-h1080")
                if len(landscape) >= max_photos:
                    break
    return (landscape + other)[:max_photos]


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    v = value & 0xFFFFFFFFFFFFFFFF
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    res = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, pos
        shift += 7
    raise ValueError("truncated varint")


def encode_cast_message(src: str, dst: str, namespace: str, payload: str) -> bytes:
    """Encode a length-prefixed Google Cast v2 protobuf packet."""
    sb, db, nb, pb = src.encode(), dst.encode(), namespace.encode(), payload.encode()
    body = (
        b"\x08\x00"
        + b"\x12"
        + _encode_varint(len(sb))
        + sb
        + b"\x1a"
        + _encode_varint(len(db))
        + db
        + b"\x22"
        + _encode_varint(len(nb))
        + nb
        + b"\x28\x00"
        + b"\x32"
        + _encode_varint(len(pb))
        + pb
    )
    return struct.pack(">I", len(body)) + body


def decode_cast_message_body(body: bytes) -> tuple[str, str, str, str]:
    """Decode ``(source_id, destination_id, namespace, payload_utf8)``."""
    pos = 0
    src = dst = ns = payload = ""
    while pos < len(body):
        tag, pos = _decode_varint(body, pos)
        field_no, wire = tag >> 3, tag & 0x07
        if wire == 0:
            _, pos = _decode_varint(body, pos)
        elif wire == 2:
            ln, pos = _decode_varint(body, pos)
            raw = body[pos : pos + ln]
            pos += ln
            if field_no == 2:
                src = raw.decode("utf-8", errors="replace")
            elif field_no == 3:
                dst = raw.decode("utf-8", errors="replace")
            elif field_no == 4:
                ns = raw.decode("utf-8", errors="replace")
            elif field_no == 6:
                payload = raw.decode("utf-8", errors="replace")
        else:
            break
    return src, dst, ns, payload


class CastV2PhotoSink:
    """Zero-dependency synchronous Google Cast v2 Default Media Receiver client."""

    NS_CONN = "urn:x-cast:com.google.cast.tp.connection"
    NS_HB = "urn:x-cast:com.google.cast.tp.heartbeat"
    NS_RECV = "urn:x-cast:com.google.cast.receiver"
    NS_MEDIA = "urn:x-cast:com.google.cast.media"
    APP_ID = "CC1AD845"

    def __init__(self, host: str, port: int = 8009) -> None:
        self.host = host.strip()
        self.port = port
        self._lock = threading.Lock()
        self._sock: ssl.SSLSocket | None = None
        self._transport_id: str | None = None
        self._session_id: str | None = None
        self._req_id = 1

    def _next_id(self) -> int:
        r = self._req_id
        self._req_id += 1
        return r

    @staticmethod
    def _recv_exact(sock: ssl.SSLSocket, count: int) -> bytes:
        buf = bytearray()
        while len(buf) < count:
            ch = sock.recv(count - len(buf))
            if not ch:
                raise EOFError("Cast TLS connection closed")
            buf.extend(ch)
        return bytes(buf)

    def _close_unlocked(self) -> None:
        sock = self._sock
        self._sock = None
        self._transport_id = None
        self._session_id = None
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()

    def _ensure_session_unlocked(self) -> tuple[ssl.SSLSocket, str]:
        if self._sock is not None and self._transport_id is not None:
            return self._sock, self._transport_id
        self._close_unlocked()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection((self.host, self.port), timeout=4.0)
        raw.settimeout(6.0)
        tls = ctx.wrap_socket(raw, server_hostname=self.host)
        self._sock = tls
        tls.sendall(
            encode_cast_message(
                "sender-0", "receiver-0", self.NS_CONN, '{"type":"CONNECT","origin":{}}'
            )
        )
        tls.sendall(
            encode_cast_message(
                "sender-0",
                "receiver-0",
                self.NS_RECV,
                json.dumps(
                    {
                        "type": "LAUNCH",
                        "appId": self.APP_ID,
                        "requestId": self._next_id(),
                    }
                ),
            )
        )
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            ln = struct.unpack(">I", self._recv_exact(tls, 4))[0]
            _s, _d, ns, pl = decode_cast_message_body(self._recv_exact(tls, ln))
            if ns == self.NS_HB and '"PING"' in pl:
                tls.sendall(
                    encode_cast_message(
                        "sender-0", "receiver-0", self.NS_HB, '{"type":"PONG"}'
                    )
                )
                continue
            if ns == self.NS_RECV and pl:
                data = json.loads(pl)
                if data.get("type") == "RECEIVER_STATUS":
                    for app in data.get("status", {}).get("applications", []):
                        if app.get("appId") == self.APP_ID and app.get("transportId"):
                            self._transport_id = str(app["transportId"])
                            self._session_id = str(app.get("sessionId") or "")
                            break
                    if self._transport_id:
                        break
        if not self._transport_id:
            self._close_unlocked()
            raise TimeoutError("Cast receiver did not return transportId")
        tls.sendall(
            encode_cast_message(
                "sender-0",
                self._transport_id,
                self.NS_CONN,
                '{"type":"CONNECT","origin":{}}',
            )
        )
        return tls, self._transport_id

    def show_photo(self, url: str, title: str, subtitle: str = "Atmosphere") -> bool:
        if not self.host or not url.startswith("https://"):
            return False
        with self._lock:
            for _ in range(2):
                try:
                    tls, tid = self._ensure_session_unlocked()
                    ctype = (
                        "image/png" if url.lower().endswith(".png") else "image/jpeg"
                    )
                    load_msg = {
                        "type": "LOAD",
                        "requestId": self._next_id(),
                        "autoplay": True,
                        "media": {
                            "contentId": url,
                            "contentType": ctype,
                            "streamType": "NONE",
                            "metadata": {
                                "metadataType": 4,
                                "title": title,
                                "subtitle": subtitle,
                            },
                        },
                    }
                    tls.sendall(
                        encode_cast_message(
                            "sender-0", tid, self.NS_MEDIA, json.dumps(load_msg)
                        )
                    )
                    return True
                except Exception:
                    self._close_unlocked()
        return False

    def stop(self) -> None:
        with self._lock:
            if self._sock is not None:
                with contextlib.suppress(Exception):
                    msg: dict[str, object] = {
                        "type": "STOP",
                        "requestId": self._next_id(),
                    }
                    if self._session_id:
                        msg["sessionId"] = self._session_id
                    self._sock.sendall(
                        encode_cast_message(
                            "sender-0", "receiver-0", self.NS_RECV, json.dumps(msg)
                        )
                    )
            self._close_unlocked()
