"""
src/replay/controller.py

Seekable replay state machine for SL2 and NMEA file replay.
Supports pause/play/seek/speed_multiplier.
On seek, calls on_seek callback so orchestrator can reset Kalman state.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, List, Any


@dataclass
class ReplayState:
    total_packets: int
    current_packet: int = 0
    paused: bool = False
    speed_multiplier: float = 1.0

    @property
    def position_fraction(self) -> float:
        if self.total_packets == 0:
            return 0.0
        return self.current_packet / self.total_packets

    @property
    def duration_s(self) -> float:
        return self.total_packets * self._packet_interval_s if hasattr(self, '_packet_interval_s') else 0.0


class ReplayController:
    """
    Owns the read cursor for an SL2 or NMEA file.
    Emits decoded SensorFrames onto a queue at the original capture rate.

    Usage:
        ctrl = ReplayController(packets, packet_interval_s=0.2, queue=raw_q, decode_fn=decode_sl2_packet)
        asyncio.create_task(ctrl.run())
        # From WebSocket handler:
        await ctrl.seek(0.4)
        ctrl.pause()
        ctrl.play()
    """

    def __init__(
        self,
        packets: List[Any],
        packet_interval_s: float,
        queue: asyncio.Queue,
        decode_fn: Callable,
        on_seek: Optional[Callable] = None,
    ):
        self._packets = packets
        self._interval = packet_interval_s
        self._queue = queue
        self._decode = decode_fn
        self._on_seek = on_seek
        self._state = ReplayState(total_packets=len(packets))
        self._state._packet_interval_s = packet_interval_s

        self._seek_event = asyncio.Event()
        self._seek_target: Optional[int] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # starts unpaused

    # ── Public control API (called from WebSocket handler) ───────────────────

    def pause(self):
        self._pause_event.clear()
        self._state.paused = True

    def play(self):
        self._pause_event.set()
        self._state.paused = False

    async def seek(self, fraction: float):
        """Seek to fraction [0.0, 1.0] of the recording."""
        target = int(fraction * self._state.total_packets)
        target = max(0, min(self._state.total_packets - 1, target))
        self._seek_target = target
        self._seek_event.set()

    def set_speed(self, multiplier: float):
        self._state.speed_multiplier = max(0.1, min(10.0, multiplier))

    @property
    def state(self) -> ReplayState:
        return self._state

    @property
    def duration_s(self) -> float:
        return self._state.total_packets * self._interval

    def get_position_s(self) -> float:
        return self._state.current_packet * self._interval

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self):
        i = 0
        while True:
            # Loop back to start when replay ends (no map reset — keep accumulating)
            if i >= len(self._packets):
                i = 0

            # Handle seek
            if self._seek_event.is_set():
                self._seek_event.clear()
                i = self._seek_target
                self._seek_target = None
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if self._on_seek is not None:
                    self._on_seek()

            # Handle pause
            await self._pause_event.wait()

            self._state.current_packet = i
            frame = self._decode(self._packets[i])
            if frame is not None:
                await self._queue.put(frame)
            i += 1

            delay = self._interval / self._state.speed_multiplier
            await asyncio.sleep(delay)
