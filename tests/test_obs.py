import asyncio
import base64
import types

import cv2
import numpy as np
import pytest

from pokesim.api.server import Live, obs_backoff
from pokesim.vision.obs_client import ObsFrames


def jpeg() -> str:
    ok, buf = cv2.imencode(".jpg", np.zeros((4, 4, 3), np.uint8))
    return "data:image/jpg;base64," + base64.b64encode(buf.tobytes()).decode()


class Client:
    def __init__(self, fail: bool):
        self.fail, self.disconnected = fail, False

    def get_source_screenshot(self, *args):
        if self.fail:
            raise ConnectionError("socket is already closed.")
        return types.SimpleNamespace(image_data=jpeg())

    def disconnect(self):
        self.disconnected = True


def test_a_closed_connection_is_replaced_on_the_next_grab():
    made = []

    def connect():
        if len(made) == 1:  # OBS was closed: connecting fails once
            made.append(None)
            raise ConnectionRefusedError("OBS not running")
        made.append(Client(fail=not made))
        return made[-1]

    frames = ObsFrames("Capture", connect=connect)
    with pytest.raises(ConnectionError):
        frames.grab()  # the first connection dies
    assert frames.client is None and made[0].disconnected
    with pytest.raises(ConnectionRefusedError):
        frames.grab()  # reconnecting fails while OBS is down
    frame = frames.grab()  # OBS is back
    assert frame.shape == (4, 4, 3) and len(made) == 3


def test_backoff_grows_and_is_capped():
    assert [obs_backoff(n) for n in (1, 2, 3, 4, 5, 6, 7, 20)] == [3, 3, 3, 6, 12, 24, 30, 30]


def test_obs_status_is_shown_on_changes_and_logged_once_a_minute(capsys):
    sent = []

    async def broadcast():
        sent.append(dict(live.obs))

    live = types.SimpleNamespace(obs={"connected": True, "since": 0.0, "failures": 0, "error": None},
                                 broadcast=broadcast)

    async def run():
        for _ in range(5):
            await Live.obs_failed(live, ConnectionError("closed"))
        await Live.obs_connected(live)

    asyncio.run(run())
    out = capsys.readouterr().out
    assert out.count("OBS grab failed") == 1  # not once per retry
    assert "reconnected after 5 failed attempts" in out
    # broadcast when it went down and when it came back, not on every retry
    assert [s["connected"] for s in sent] == [False, True]
    assert live.obs["failures"] == 0 and live.obs["error"] is None
