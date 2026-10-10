"""Run a private local Showdown server (e.g. for training), restartable to bound its memory growth."""
import os
import signal
import subprocess
import time

import httpx
from poke_env import ServerConfiguration

from pokesim.config import SHOWDOWN_DIR


class ShowdownServer:
    def __init__(self, port: int = 8001):
        self.port = port
        self.proc: subprocess.Popen | None = None

    @property
    def configuration(self) -> ServerConfiguration:
        return ServerConfiguration(f"ws://localhost:{self.port}/showdown/websocket",
                                   "https://play.pokemonshowdown.com/action.php?")

    def start(self, timeout: float = 60.0) -> "ShowdownServer":
        self.proc = subprocess.Popen(["node", "pokemon-showdown", "start", "--no-security", str(self.port)],
                                     cwd=SHOWDOWN_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                httpx.get(f"http://localhost:{self.port}", timeout=1)
                return self
            except httpx.HTTPError:
                time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"Showdown did not start on port {self.port}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)  # the server forks worker processes
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc = None

    def __enter__(self) -> "ShowdownServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
