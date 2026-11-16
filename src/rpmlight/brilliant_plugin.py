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
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from rpmlight.engine import describe_slider_setting
from rpmlight.sinks import KodiJsonRpcToastSink, MiniMqttPublisher


@dataclass
class BrilliantAmbiPluginState:
    """Thread-safe state shared between the Brilliant Thrift bus host and the capture loop."""

    display_name: str = "TV AmbiLED"
    peripheral_id: str = "rpmlight_tv_ambiled"
    room_id: str = ""
    enabled: bool = False
    slider_pct: int = 25
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
    _lock_timer: threading.Timer | None = field(default=None, init=False, repr=False)

    def cancel_pending_lock(self) -> None:
        if self._lock_timer is not None:
            try:
                self._lock_timer.cancel()
            except Exception:
                pass
            self._lock_timer = None

    def schedule_slider_lock_in(self, pct: int, delay_s: float = 2.0) -> None:
        self.cancel_pending_lock()
        if self._toast_timer is not None:
            try:
                self._toast_timer.cancel()
            except Exception:
                pass
            self._toast_timer = None
        pct_clamped = max(1, min(100, int(pct)))

        def _lock_in() -> None:
            self._lock_timer = None
            if not self.enabled:
                return
            self.slider_pct = pct_clamped
            if self.on_change is not None:
                self.on_change(True, pct_clamped)
            self.schedule_status_toast(0.05)

        self._lock_timer = threading.Timer(delay_s, _lock_in)
        self._lock_timer.daemon = True
        self._lock_timer.start()

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


def build_bus_push_handlers(
    state: BrilliantAmbiPluginState,
    set_var: Callable[[str, object], Awaitable[None]],
    *,
    is_host_ready: Callable[[], bool] = lambda: True,
    lock_in_delay_s: float = 2.0,
) -> tuple[Callable[..., Awaitable[bool]], Callable[..., Awaitable[bool]]]:
    """Create ``(on_push_on, on_push_intensity)`` handlers for the hosted Brilliant peripheral.

    Handles both bare wall-switch ON taps (defaulting cleanly to 25%) and explicit
    Home Assistant ``light.turn_on(brightness_pct=X)`` commands (where ``on=1`` is
    immediately followed by an explicit ``intensity != pre_on_intensity``).
    """
    current_bus_intensity = state.slider_pct * 10
    pre_on_intensity = current_bus_intensity
    turn_on_window_until = 0.0
    reset_25_task: asyncio.Task[None] | None = None
    dormant_reset_task: asyncio.Task[None] | None = None

    def _cancel_task(t: asyncio.Task[None] | None) -> None:
        if t is not None and not t.done():
            t.cancel()

    async def _delayed_reset_25() -> None:
        nonlocal current_bus_intensity
        try:
            await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            return
        if not state.enabled:
            return
        current_bus_intensity = 250
        await set_var("intensity", 250)
        state.slider_pct = 25
        if state.on_change is not None:
            state.on_change(True, 25)
        state.schedule_status_toast(0.20)

    async def _dormant_reset_intensity_250() -> None:
        """Reset dormant bus intensity to 250 after turning OFF so next bare ON starts at 25%."""
        nonlocal current_bus_intensity
        try:
            await asyncio.sleep(0.30)
        except asyncio.CancelledError:
            return
        if state.enabled:
            return
        current_bus_intensity = 250
        await set_var("intensity", 250)

    def _extract_arg(args: tuple[object, ...]) -> object:
        # Cython VariableSpec push_func may pass (val,) or (self, val)
        return args[-1] if args else None

    async def on_push_on(*args: object) -> bool:
        nonlocal \
            pre_on_intensity, \
            turn_on_window_until, \
            reset_25_task, \
            dormant_reset_task
        val = _extract_arg(args)
        on_int = 1 if str(val).strip() in ("1", "True", "true") else 0
        await set_var("on", on_int)
        if not is_host_ready():
            state.enabled = bool(on_int)
            return True
        if on_int == 1 and not state.enabled:
            _cancel_task(dormant_reset_task)
            _cancel_task(reset_25_task)
            state.cancel_pending_lock()
            pre_on_intensity = current_bus_intensity
            turn_on_window_until = time.monotonic() + 0.60
            state.enabled = True
            state.slider_pct = 25
            if state.on_change is not None:
                state.on_change(True, 25)
            reset_25_task = asyncio.create_task(_delayed_reset_25())
        elif on_int == 0 and state.enabled:
            _cancel_task(reset_25_task)
            _cancel_task(dormant_reset_task)
            turn_on_window_until = 0.0
            state.cancel_pending_lock()
            state.enabled = False
            if state.on_change is not None:
                state.on_change(False, state.slider_pct)
            state.schedule_status_toast(0.25)
            dormant_reset_task = asyncio.create_task(_dormant_reset_intensity_250())
        return True

    async def on_push_intensity(*args: object) -> bool:
        nonlocal \
            current_bus_intensity, \
            turn_on_window_until, \
            reset_25_task, \
            dormant_reset_task
        val = _extract_arg(args)
        try:
            inten = int(float(str(val).strip()))
        except Exception:
            return False
        await set_var("intensity", inten)
        if not is_host_ready():
            current_bus_intensity = inten
            return True

        now = time.monotonic()
        if now < turn_on_window_until:
            # Ignore stale pre-off intensity echo or our own 250 reset,
            # EXCEPT when an explicit new target brightness arrives (e.g. HA light.turn_on(brightness_pct=60)).
            if inten in (pre_on_intensity, 250) or inten <= 0:
                return True
            _cancel_task(reset_25_task)
            turn_on_window_until = 0.0

        current_bus_intensity = inten
        if inten <= 0:
            _cancel_task(reset_25_task)
            _cancel_task(dormant_reset_task)
            state.cancel_pending_lock()
            state.enabled = False
            asyncio.create_task(set_var("on", 0))
            if state.on_change is not None:
                state.on_change(False, state.slider_pct)
            state.schedule_status_toast(0.25)
            dormant_reset_task = asyncio.create_task(_dormant_reset_intensity_250())
        else:
            _cancel_task(dormant_reset_task)
            pct = max(1, min(100, round(inten / 10.0)))
            was_off = not state.enabled
            state.enabled = True
            state.schedule_slider_lock_in(pct, delay_s=lock_in_delay_s)
            if was_off:
                asyncio.create_task(set_var("on", 1))
        return True

    return on_push_on, on_push_intensity


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
        state._update_fn = Peripheral.__dict__["_set_value_internal"]
        flags = gflags.FLAGS
        try:
            _ = flags.message_bus_server_socket_path
        except Exception:
            flags([""])
        flags.message_bus_server_socket_path = socket_path

        while True:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state._loop = loop

            async def _async_host(
                host_loop: asyncio.AbstractEventLoop = loop,
            ) -> None:
                host_ready = False

                async def _set_var(name: str, val: object) -> None:
                    if state._inst is not None and state._update_fn is not None:
                        try:
                            res = state._update_fn(state._inst, name, val, notify=True)  # type: ignore[operator]
                            if inspect.isawaitable(res):
                                await res
                        except Exception:
                            pass

                on_push_on, on_push_intensity = build_bus_push_handlers(
                    state,
                    _set_var,
                    is_host_ready=lambda: host_ready,
                )

                def make_push_passthrough(var_name: str) -> object:
                    async def _push(*args: object) -> bool:
                        val = args[-1] if args else None
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
                            int,
                            True,
                            default_value=state.slider_pct * 10,
                            push_func=on_push_intensity,
                        ),
                        "dimmable": VariableSpec(int, False, default_value=1),
                        "max_intensity_value": VariableSpec(
                            int, False, default_value=1000
                        ),
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
                        "mode_transition_settings": VariableSpec(
                            str,
                            True,
                            default_value="{}",
                            push_func=make_push_passthrough("mode_transition_settings"),
                        ),
                    }

                class TVAmbiLightPeripheral(Peripheral):  # type: ignore[misc]
                    def __init__(self, *args: object, **kwargs: object) -> None:
                        super().__init__(*args, **kwargs)
                        state._inst = self

                    @property
                    def name(self) -> str:
                        return state.display_name

                    @property
                    def peripheral_type(self) -> int:
                        return 27  # PeripheralType.LIGHT

                    def _my_variables(self) -> dict[str, object]:
                        return build_vars()

                cfg = PeripheralConfig(state.peripheral_id, TVAmbiLightPeripheral)
                spec = HostedStartableSpec(state.peripheral_id, cfg, {})
                host = PeripheralHost(
                    loop=host_loop,
                    startable_id=state.peripheral_id,
                    startables_to_host=[spec],
                    parallel_registration_limit=1,
                    raise_errors_for_lost_user_configured_data=False,
                    message_bus_address_override=None,
                )
                await host.start()
                await _set_var("on", 1 if state.enabled else 0)
                await _set_var("intensity", state.slider_pct * 10)
                await asyncio.sleep(0.5)
                host_ready = True
                while True:
                    await asyncio.sleep(3600)

            try:
                loop.run_until_complete(_async_host())
            except Exception as exc:
                print(
                    f"[RPMLight] Brilliant Thrift host retrying in 2s after: {exc}",
                    flush=True,
                )
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
            time.sleep(2.0)

    t = threading.Thread(
        target=_thread_main, name="RPMLight-BrilliantHost", daemon=True
    )
    t.start()
    return True
