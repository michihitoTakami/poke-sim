"""Frames from OBS (obs-websocket v5) and team preview detection.

`ObsFrames` grabs the capture source as a 1920x1080 image a few times per second (GetSourceScreenshot).
`is_team_preview` recognises the team preview screen: the opponent's 6 crimson cards in the right column and the
pink name banner above them.
`PreviewWatcher` turns a stream of frames into one "team preview seen" event per battle.

OBS: Tools -> WebSocket Server Settings -> enable, note port/password; the source is the capture card's source name.
"""
import base64
import time
from collections.abc import Callable

import cv2
import numpy as np

from pokesim.vision.preview import CARD_X0, CARD_X1, card_mask, find_cards, normalize


def is_team_preview(frame: np.ndarray) -> bool:
    """The left margins of the opponent cards are card-coloured (at least 4 of 6: with the strength display on, a
    warning icon covers part of some margins), and the pink name banner sits above them."""
    img = normalize(frame)
    mask = card_mask(img)
    if sum(mask[y0 + 10:y1 - 10, CARD_X0 + 6:CARD_X0 + 50].mean() >= 0.5 for y0, y1 in find_cards(img)) < 4:
        return False
    hsv = cv2.cvtColor(img[100:140, CARD_X0 + 20:CARD_X1 - 20], cv2.COLOR_BGR2HSV)
    banner = ((hsv[..., 0] >= 158) & (hsv[..., 0] <= 178) & (hsv[..., 1] > 110) & (hsv[..., 2] > 140)).mean()
    return banner > 0.4


class ObsFrames:
    """Connects lazily and again after any failure: OBS can be closed and reopened while the server runs (a
    connection, once closed, never recovers by itself)."""

    def __init__(self, source: str, host: str = "localhost", port: int = 4455, password: str = "",
                 connect: Callable | None = None):
        self.source, self.host, self.port, self.password = source, host, port, password
        self._connect = connect or self._obs_client
        self.client = None

    def _obs_client(self):
        import obsws_python as obs

        return obs.ReqClient(host=self.host, port=self.port, password=self.password, timeout=5)

    def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001 - already closed
                pass

    def grab(self) -> np.ndarray | None:
        """A frame, or an exception (the next call reconnects)."""
        try:
            if self.client is None:
                self.client = self._connect()
            resp = self.client.get_source_screenshot(self.source, "jpg", 1920, 1080, 90)  # ~15 ms; png ~70-400 ms
        except Exception:
            self.close()
            raise
        data = base64.b64decode(resp.image_data.split(",", 1)[-1])
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class PreviewWatcher:
    """Calls `on_preview(frame)` once when the team preview has been seen in `confirm` consecutive frames; re-arms
    after it has been gone for `rearm` seconds (the next battle)."""

    def __init__(self, on_preview: Callable[[np.ndarray], None], confirm: int = 2, rearm: float = 20.0):
        self.on_preview, self.confirm, self.rearm = on_preview, confirm, rearm
        self.streak, self.fired, self.last_seen = 0, False, 0.0

    def feed(self, frame: np.ndarray, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if is_team_preview(frame):
            self.streak += 1
            self.last_seen = now
            if self.streak >= self.confirm and not self.fired:
                self.fired = True
                self.on_preview(frame)
                return True
        else:
            self.streak = 0
            if self.fired and now - self.last_seen > self.rearm:
                self.fired = False
        return False
