"""Brilliant Smart Home Control Panel Plugin for RPMLight.

Hosts a native ``TV AmbiLED`` dimmable light peripheral (``PeripheralType.LIGHT = 27``)
on the panel's local Thrift bus (``/var/run/brilliant/server_socket``) so:
1. The Brilliant touchscreen displays a native ``TV AmbiLED`` On/Off button and
   ``0-100%`` slider in the configured room.
2. ``brilliant-mqtt`` automatically discovers ``TV AmbiLED`` on the local bus
   and bridges it to Home Assistant with zero custom forks required.
3. Turning ``TV AmbiLED`` OFF (or receiving an MQTT ``OFF`` on the TV power topic)
   immediately closes ``/dev/video0`` (extinguishing the red camera LED), blanks
   Nanoleaf 4D, and publishes ``{"restore": true}`` to restore room lighting.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable

from rpmlight.engine import describe_slider_setting
from rpmlight.sinks import KodiJsonRpcToastSink, MiniMqttPublisher


@dataclass
class BrilliantAmbiPluginState:
    """Thread-safe state shared between the Brilliant Thrift bus host and the capture loop."""

    display_name: str = "TV AmbiLED"
    peripheral_id: str = "rpmlight_tv_ambiled"
    room_id: str = ""
    enabled: bool = False
    slider_pct: int = 50
    mqtt_pub: MiniMqttPublisher | None = None
    mqtt_status_topic: str = "rpmlight/status"
    kodi_toast: KodiJsonRpcToastSink | None = None
    on_change: Callable[[bool, int], None] | None = None

    _loop: asyncio.AbstractEventLoop | None = field(
        default=None, init=False, repr=False
    )
    _inst: object | None = field(default=None, init=False, repr=False)
    _update_fn: object | None = field(default=None, init=False, repr=False)
    _toast_timer: threading.Timer | None = field(default=None, init=False, repr=False)

    def schedule_status_toast(self, delay_s: float = 0.8) -> None:
        if self._toast_timer is not None:
            try:
                self._toast_timer.cancel()
            except Exception:
                pass

        def _emit() -> None:
            title, message = describe_slider_setting(
                self.slider_pct, is_on=self.enabled
            )
            if self.mqtt_pub is not None and self.mqtt_status_topic:
                payload = json.dumps(
                    {
                        "state": "ON" if self.enabled else "OFF",
                        "slider_pct": self.slider_pct if self.enabled else 0,
                        "title": title,
                        "message": message,
                        "summary": f"{title} — {message}",
                    }
                )
                self.mqtt_pub.publish(self.mqtt_status_topic, payload)
            if self.kodi_toast is not None:
                self.kodi_toast.send_toast(title, message)

        self._toast_timer = threading.Timer(delay_s, _emit)
        self._toast_timer.daemon = True
        self._toast_timer.start()

    def sync_to_brilliant_ui(self, on_val: int, intensity_val: int) -> None:
        """Push updated ``on`` and ``intensity`` back to the Brilliant touchscreen UI."""
        loop = self._loop
        inst = self._inst
        upd = self._update_fn
        if loop is None or inst is None or upd is None:
            return

        async def _push_ui() -> None:
            try:
                res1 = upd(inst, "on", int(on_val), notify=True)  # type: ignore[operator]
                if inspect.isawaitable(res1):
                    await res1
                res2 = upd(inst, "intensity", int(intensity_val), notify=True)  # type: ignore[operator]
                if inspect.isawaitable(res2):
                    await res2
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_push_ui(), loop)
        except Exception:
            pass


def start_brilliant_virtual_light(
    state: BrilliantAmbiPluginState,
    socket_path: str = "/var/run/brilliant/server_socket",
) -> bool:
    """Start background thread hosting the native ``TV AmbiLED`` light on a Brilliant panel.

    Returns ``True`` if the Brilliant firmware libraries were found and the host thread
    was started, or ``False`` when running on a non-Brilliant Linux machine.
    """
    try:
        import gflags  # type: ignore[import-not-found]
        from lib.startables.startable import HostedStartableSpec  # type: ignore[import-not-found]
        from peripherals.lib.peripheral_service.peripheral import (  # type: ignore[import-not-found]
            Peripheral,
            VariableSpec,
        )
        from peripherals.lib.peripheral_service.peripheral_host import (  # type: ignore[import-not-found]
            PeripheralConfig,
            PeripheralHost,
        )
        from thrift_types.configuration.ttypes import RoomAssignment  # type: ignore[import-not-found]
    except ImportError:
        return False

    def _thread_main() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state._loop = loop
        state._update_fn = Peripheral.__dict__["_set_value_internal"]

        flags = gflags.FLAGS
        try:
            _ = flags.message_bus_server_socket_path
        except Exception:
            flags([""])
        flags.message_bus_server_socket_path = socket_path

        async def _async_host() -> None:
            host_ready = False

            async def _set_var(name: str, val: object) -> None:
                if state._inst is not None and state._update_fn is not None:
                    try:
                        res = state._update_fn(state._inst, name, val, notify=True)  # type: ignore[operator]
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        pass

            async def _delayed_reset_50() -> None:
                await asyncio.sleep(0.10)
                await _set_var("intensity", 500)
                state.slider_pct = 50
                if state.on_change is not None:
                    state.on_change(state.enabled, state.slider_pct)

            async def on_push_on(val: object) -> bool:
                on_int = 1 if str(val).strip() in ("1", "True", "true") else 0
                await _set_var("on", on_int)
                if not host_ready:
                    state.enabled = bool(on_int)
                    return True
                if on_int == 1 and not state.enabled:
                    state.enabled = True
                    state.slider_pct = 50
                    if state.on_change is not None:
                        state.on_change(True, 50)
                    asyncio.create_task(_delayed_reset_50())
                    state.schedule_status_toast(0.25)
                elif on_int == 0 and state.enabled:
                    state.enabled = False
                    if state.on_change is not None:
                        state.on_change(False, state.slider_pct)
                    state.schedule_status_toast(0.25)
                return True

            async def on_push_intensity(val: object) -> bool:
                try:
                    inten = int(float(str(val).strip()))
                except Exception:
                    return False
                await _set_var("intensity", inten)
                if not host_ready:
                    return True
                if inten <= 0:
                    state.enabled = False
                    asyncio.create_task(_set_var("on", 0))
                    if state.on_change is not None:
                        state.on_change(False, state.slider_pct)
                    state.schedule_status_toast(0.25)
                else:
                    pct = max(1, min(100, round(inten / 10.0)))
                    was_off = not state.enabled
                    state.enabled = True
                    state.slider_pct = pct
                    if state.on_change is not None:
                        state.on_change(True, pct)
                    state.schedule_status_toast(1.0)
                    if was_off:
                        asyncio.create_task(_set_var("on", 1))
                return True

            def make_push_passthrough(var_name: str) -> object:
                async def _push(val: object) -> bool:
                    await _set_var(var_name, val)
                    return True

                return _push

            room_ids = [state.room_id] if state.room_id else []

            def build_vars() -> dict[str, object]:
                return {
                    "on": VariableSpec(
                        int,
                        True,
                        default_value=1 if state.enabled else 0,
                        push_func=on_push_on,
                    ),
                    "intensity": VariableSpec(
                        int, True, default_value=500, push_func=on_push_intensity
                    ),
                    "dimmable": VariableSpec(int, False, default_value=1),
                    "max_intensity_value": VariableSpec(int, False, default_value=1000),
                    "minimum_dim_level": VariableSpec(
                        int,
                        True,
                        default_value=10,
                        push_func=make_push_passthrough("minimum_dim_level"),
                    ),
                    "maximum_dim_level": VariableSpec(
                        int,
                        True,
                        default_value=1000,
                        push_func=make_push_passthrough("maximum_dim_level"),
                    ),
                    "display_name": VariableSpec(
                        str,
                        True,
                        default_value=state.display_name,
                        push_func=make_push_passthrough("display_name"),
                    ),
                    "room_assignment": VariableSpec(
                        RoomAssignment,
                        True,
                        default_value=RoomAssignment(room_ids=room_ids),
                        push_func=make_push_passthrough("room_assignment"),
                    ),
                }

            class TVAmbiLightPeripheral(Peripheral):  # type: ignore[misc]
                def __init__(self, *args: object, **kwargs: object) -> None:
                    super().__init__(*args, **kwargs)
                    state._inst = self

                def _my_variables(self) -> dict[str, object]:
                    return build_vars()

            spec = HostedStartableSpec(
                state.peripheral_id,
                PeripheralConfig(state.peripheral_id, TVAmbiLightPeripheral),
                {"peripheral_type": 27},  # PeripheralType.LIGHT
            )
            host = PeripheralHost(
                loop=loop,
                startable_id=f"rpmlight_host_{os.getpid()}",
                startables_to_host=[spec],
                parallel_registration_limit=1,
                raise_errors_for_lost_user_configured_data=False,
                message_bus_address_override=None,
            )
            await host.start()
            await _set_var("on", 1 if state.enabled else 0)
            await _set_var("intensity", state.slider_pct * 10)
            host_ready = True
            while True:
                await asyncio.sleep(3600)

        loop.run_until_complete(_async_host())

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()
    return True
