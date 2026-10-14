"""Live battle support server: team preview -> lineup suggestion -> per-turn advice -> record.

Pages: /            control page (recognition results, lineup table, turn input, advice)
       /overlay     transparent overlay for an OBS browser source (advice only, updated over the WebSocket)
API:   POST /api/preview/recognize {image: base64}   -> recognised opponent 6 with candidates
       POST /api/preview {species: [6]}               -> opponent pick probabilities and our lineup ranking
       POST /api/select {lineup: [3 species]}         -> starts the battle with that lineup (lead first)
       POST /api/events {events: [...], force_switch}  -> applies turn events (pokesim.live.events), returns advice
       POST /api/lines {lines: [...]}                  -> raw protocol lines (for tools)
       POST /api/end {result: "win"|"loss"}            -> saves the record
       GET  /api/state, /api/options/...              WS /ws (state pushes)

Usage: uv run python -m pokesim.api.server --policy checkpoints/ppo_baton_v2.zip [--port 8100]
         [--obs-source "Capture Card"]  (password from $OBS_PASSWORD)   # watch OBS: team preview, then the battle
With OBS, the battle screen is read during the game (pokesim.vision.battle + pokesim.live.screen): messages, HP and
the command prompt become turn events and advice without manual input (--no-screen to turn it off).
For practice, reviewing games and commentary: official tournaments forbid outside help during battles.
"""
import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from pokesim.config import RAW_DIR, ROOT
from pokesim.live.advisor import Advisor, Suggestion
from pokesim.live.events import EventTranslator
from pokesim.live.screen import ScreenTracker, Vocab
from pokesim.live.selection import MatchupModel, PickModel
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id

WEB = ROOT / "web"
LOG_DIR = RAW_DIR / "live/logs"


def ja_names() -> dict[str, str]:
    try:
        from pokesim.rl.advise import ja_names as load

        return load()
    except Exception:  # noqa: BLE001 - no DB: English ids only
        return {}


class Live:
    def __init__(self, team: Path, policy: str | None, lookahead: bool = True):
        self.sets = json.loads(Path(team).read_text())
        self.roster = [s["species"] for s in self.sets]
        self.ja = ja_names()
        self.advisor = Advisor(self.sets, policy=policy, lookahead=lookahead)
        self.picks_model, self.matchups = PickModel(), MatchupModel()
        self.reader = None  # PreviewReader, loaded on first use (GPU model)
        self.screen = None  # ScreenReader (OCR), loaded on the first battle frame
        self.read_screen = True
        self.clients: set[WebSocket] = set()
        self.reset()

    def reset(self) -> None:
        self.session: BattleSession | None = None
        self.translator: EventTranslator | None = None
        self.preview: dict | None = None
        self.lineup: list[str] | None = None
        self.suggestion: Suggestion | None = None
        self.history: list[dict] = []
        self.recognized: list[dict] | None = None
        self.started = time.time()
        self.tracker: ScreenTracker | None = None
        self.seen_me: list[str] = []      # our Pokémon that have been on the field
        self.screen_log: list[str] = []   # what the screen reader understood, for the control page
        self.saved: str | None = None

    # --- names ---------------------------------------------------------------------------------------------------
    def name(self, sid: str) -> str:
        return self.ja.get(to_id(sid), sid)

    def label(self, label: str) -> str:
        kind, _, rest = label.partition(":")
        if kind == "switch":
            return f"交代→{self.name(rest)}"
        move, _, mega = rest.partition(":")
        return self.name(move) + ("（メガシンカ）" if mega else "")

    # --- state -----------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        out: dict = {"preview": self.preview, "recognized": self.recognized, "lineup": self.lineup, "advice": None,
                     "battle": None, "screen": {"on": self.tracker is not None and self.read_screen,
                                                "log": self.screen_log[-25:], "saved": self.saved}}
        if self.session:
            b = self.session.battle
            me, opp = b.active_pokemon, b.opponent_active_pokemon

            def mon(m, own):
                if m is None:
                    return None
                return {"species": m.species, "name": self.name(m.species), "hp": round(m.current_hp_fraction, 3),
                        "hp_points": m.current_hp if own else None, "status": m.status.name if m.status else None,
                        "boosts": {k: v for k, v in m.boosts.items() if v}, "moves": list(m.moves),
                        "item": m.item, "ability": m.ability, "fainted": m.fainted}
            out["battle"] = {"turn": b.turn, "me": mon(me, True), "opp": mon(opp, False),
                             "team": [mon(m, True) for m in b.team.values()],
                             "opp_team": [mon(m, False) for m in b.opponent_team.values()],
                             "finished": b.finished, "log": self.session.log[-30:]}
        if self.suggestion:
            s = self.suggestion
            out["advice"] = {"actions": [{"id": a, "label": self.label(a), "p": round(p, 3)} for a, p in s.actions[:6]],
                             "opponent": [{"id": a, "label": self.label(a), "p": round(p, 3)}
                                          for a, p in s.opponent[:6]],
                             "win_prob": s.win_prob, "elapsed": round(s.elapsed, 2)}
        return out

    async def broadcast(self) -> None:
        msg = json.dumps(self.state(), ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001 - drop dead sockets
                self.clients.discard(ws)

    async def recognize(self, frame: np.ndarray) -> list[dict]:
        if self.reader is None:
            from pokesim.vision.preview import PreviewReader

            self.reader = await asyncio.to_thread(PreviewReader)
        slots = await asyncio.to_thread(self.reader.read, frame)
        self.recognized = [{"species": s.species, "types": s.types, "gender": s.gender,
                            "candidates": [{"name": n, "ja": self.name(n), "score": round(v, 3)}
                                           for n, v in s.candidates]} for s in slots]
        return self.recognized

    async def start_preview(self, species: list[str]) -> dict:
        recognized = self.recognized
        self.reset()
        self.recognized = recognized
        picks = self.picks_model.predict(species)
        ranking = self.matchups.rank(self.roster, picks)
        self.preview = {"species": species, "names": [self.name(s) for s in species],
                        "brought": [round(float(p), 3) for p in picks.brought],
                        "lead": [round(float(p), 3) for p in picks.lead],
                        "lineups": [{"lineup": names, "names": [self.name(n) for n in names], "win": round(p, 3)}
                                    for names, p in ranking[:10]]}
        self.session = BattleSession(self.sets)
        self.session.preview(species)
        self.translator = EventTranslator(self.session)
        self.tracker = ScreenTracker(Vocab.build(self.sets, species, self.ja),
                                     maxhp={i["species"]: i["maxhp"] for i in self.session.info})
        await self.broadcast()
        return self.preview

    def set_lineup(self, lineup: list[str]) -> None:
        ids = [to_id(s) for s in lineup]
        order = [next(i for i, s in enumerate(self.sets) if to_id(s["species"]) == sid) for sid in ids]
        self.session.select(order)
        self.lineup = [self.sets[i]["species"] for i in order]

    def ensure_member(self, species: str) -> None:
        """Our Pokémon switched in: register the lineup if nobody did (best suggested lineup with this lead), or
        correct it when this one is not in it (it replaces a member not seen yet)."""
        if to_id(species) not in map(to_id, self.seen_me):
            self.seen_me.append(species)
        if self.lineup is None:
            suggested = next((entry["lineup"] for entry in (self.preview or {}).get("lineups", [])
                              if to_id(entry["lineup"][0]) == to_id(species)), None)
            self.set_lineup(suggested or [species] + [s for s in self.roster if to_id(s) != to_id(species)][:2])
        elif to_id(species) not in map(to_id, self.lineup):
            lineup = list(self.lineup)
            unseen = [i for i, s in enumerate(lineup) if to_id(s) not in map(to_id, self.seen_me)]
            lineup[unseen[-1] if unseen else -1] = species
            self.set_lineup(lineup)

    def describe(self, ev: dict) -> str:
        who = "相手" if ev.get("side") == "opp" else "自分"
        kind = ev["type"]
        if kind == "switch":
            return f"{who}: {self.name(ev['species'])} が登場"
        if kind == "move":
            return f"{who}: {self.name(ev['move'])}"
        if kind == "hp":
            return f"{who}: HP {ev['hp']}{'%' if ev.get('side') == 'opp' else ''}"
        if kind == "boost":
            return f"{who}: {ev['stat']} {'+' if ev['amount'] > 0 else ''}{ev['amount']}"
        if kind in ("turn", "force_switch"):
            return "── 次のターン ──" if kind == "turn" else "── 交代先を選ぶ ──"
        detail = {k: v for k, v in ev.items() if k not in ("type", "side")}
        return f"{who}: {kind} {detail}"

    async def on_screen_events(self, events: list[dict]) -> None:
        """Events from the screen reader -> battle state; advice at our decisions; record at the end."""
        for ev in events:
            self.screen_log.append(self.describe(ev))
            try:
                if ev["type"] == "switch" and ev["side"] == "me":
                    self.ensure_member(ev["species"])
                if ev["type"] == "force_switch":
                    await self.advise(force_switch=True)
                    continue
                self.translator.apply([{k: v for k, v in ev.items() if k != "missed"}])
                if ev["type"] == "turn" and not ev.get("missed"):
                    await self.advise()
                if ev["type"] == "win" and self.saved is None:
                    self.saved = str(save_record(self, "win" if ev["winner"] == "me" else "loss"))
            except Exception as e:  # noqa: BLE001 - one bad read must not stop the battle
                self.screen_log.append(f"(反映できませんでした: {ev['type']} {repr(e)[:80]})")
                print("screen event failed:", ev, repr(e)[:200], flush=True)
        if events:
            await self.broadcast()

    async def watch_obs(self, frames, fps: float, record: Path | None = None) -> None:
        """Poll OBS; on a newly seen team preview, recognise it and start the lineup suggestion.
        With `record`, also save frames that differ from the last saved one (screen samples for message/HP reading)."""
        from pokesim.vision.obs_client import is_team_preview

        armed, last_seen, last_saved = True, 0.0, None
        while True:
            tick = time.time()
            try:
                frame = await asyncio.to_thread(frames.grab)
            except Exception as e:  # noqa: BLE001 - OBS closed or source renamed: keep retrying
                print("OBS grab failed:", repr(e)[:120], flush=True)
                await asyncio.sleep(3)
                continue
            now = time.time()
            if record is not None and frame is not None:
                small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)
                # whole frame, or the message box / HUD rows (small text changes are a small share of the frame)
                diff = None if last_saved is None else np.abs(small - last_saved)
                if diff is None or diff.mean() > 4 or diff[38:52].mean() > 3 or diff[:9].mean() > 3:
                    last_saved = small
                    day = record / time.strftime("%Y%m%d")
                    day.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(cv2.imwrite, str(day / f"{now:.1f}.jpg"), frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            preview = frame is not None and await asyncio.to_thread(is_team_preview, frame)
            if frame is not None and not preview and self.read_screen and self.tracker is not None \
                    and not self.tracker.finished:
                if self.screen is None:
                    from pokesim.vision.battle import ScreenReader

                    self.screen = await asyncio.to_thread(ScreenReader)
                reading = await asyncio.to_thread(self.screen.read, frame)
                await self.on_screen_events(self.tracker.feed(reading))
            if preview:
                last_seen = now
                if armed:
                    armed = False
                    slots = await self.recognize(frame)
                    species = [s["species"] or (s["candidates"][0]["name"] if s["candidates"] else "") for s in slots]
                    if all(species):
                        await self.start_preview(species)
                    print("team preview:", species, flush=True)
            elif not armed and now - last_seen > 20:
                armed = True  # gone for a while: the next team preview is a new battle
            await asyncio.sleep(max(0.0, 1 / fps - (time.time() - tick)))

    async def advise(self, force_switch: bool = False) -> None:
        battle = self.session.decision(force_switch=force_switch)
        if battle.finished:
            self.suggestion = None
            return
        self.suggestion = await asyncio.to_thread(self.advisor.advise, battle, self.session.last_move(),
                                                 list(self.session.log))
        self.history.append({"turn": battle.turn, "force_switch": force_switch,
                             "actions": self.suggestion.actions[:5], "opponent": self.suggestion.opponent[:5],
                             "win_prob": self.suggestion.win_prob})


def create_app(live: Live, obs: dict | None = None) -> FastAPI:
    app = FastAPI(title="poke-sim live")

    if obs:
        @app.on_event("startup")
        async def start_obs():
            from pokesim.vision.obs_client import ObsFrames

            frames = ObsFrames(obs["source"], obs["host"], obs["port"], obs["password"])
            app.state.obs_task = asyncio.create_task(live.watch_obs(frames, obs["fps"], obs["record"]))

    @app.get("/")
    def control():
        return FileResponse(WEB / "control.html")

    @app.get("/overlay")
    def overlay():
        return FileResponse(WEB / "overlay.html")

    @app.get("/api/state")
    def state():
        return live.state()

    @app.get("/api/options/species")
    def species_options():
        legal = json.loads((RAW_DIR / "sprites/legal.json").read_text())
        return [{"id": s["id"], "name": s["name"], "ja": live.name(s["id"]), "types": s["types"]} for s in legal]

    @app.get("/api/options/moves")
    def move_options(species: str = ""):
        """Likely moves of an opponent species first (usage stats), then the rest of its learnset (or every move)."""
        from poke_env.data import GenData

        moves = GenData.from_gen(9).moves
        likely, pool = [], set(moves)
        sampler = live.advisor.engine.sampler if live.advisor.engine else None
        if species and sampler:
            base = sampler.base_id(to_id(species))
            summary = sampler.summary.get(base)
            if summary:
                likely = [m for m, _ in summary.moves.most_common(12)]
            pool = sampler.learnsets.get(base, pool) or pool
        rest = sorted(m for m in pool if m in moves and m not in likely and not moves[m].get("isZ")
                      and not moves[m].get("isMax"))
        return [{"id": m, "ja": live.name(m), "likely": m in likely} for m in likely + rest]

    @app.get("/api/team")
    def team():
        return [{"species": s["species"], "id": to_id(s["species"]), "ja": live.name(s["species"]),
                 "moves": [{"id": to_id(m), "ja": live.name(to_id(m))} for m in s["moves"]],
                 "item": s["item"]} for s in live.sets]

    @app.post("/api/preview/recognize")
    async def recognize(body: dict):
        data = base64.b64decode(body["image"].split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise HTTPException(400, "could not decode image")
        return await live.recognize(frame)

    @app.post("/api/preview")
    async def preview(body: dict):
        if len(body["species"]) != 6:
            raise HTTPException(400, "need the opponent's 6")
        return await live.start_preview(body["species"])

    @app.post("/api/select")
    async def select(body: dict):
        if live.session is None:
            raise HTTPException(400, "send the team preview first")
        live.set_lineup(body["lineup"])
        await live.broadcast()
        return live.state()

    @app.post("/api/events")
    async def events(body: dict):
        if live.translator is None:
            raise HTTPException(400, "send the team preview first")
        try:
            lines = live.translator.apply(body.get("events", []))
        except (ValueError, StopIteration, KeyError, AttributeError) as e:
            raise HTTPException(400, f"could not apply events: {e!r}") from e
        b = live.session.battle
        if body.get("advise", True) and b.active_pokemon is not None and b.opponent_active_pokemon is not None:
            await live.advise(force_switch=bool(body.get("force_switch")))
        await live.broadcast()
        return {"lines": lines, **live.state()}

    @app.post("/api/lines")
    async def lines(body: dict):
        if live.session is None:
            raise HTTPException(400, "send the team preview first")
        live.session.apply(body["lines"])
        if body.get("advise"):
            await live.advise(force_switch=bool(body.get("force_switch")))
        await live.broadcast()
        return live.state()

    @app.post("/api/end")
    async def end(body: dict):
        if live.session is None:
            raise HTTPException(400, "no battle")
        result = body.get("result")
        path = save_record(live, result)
        live.saved = str(path)
        await live.broadcast()
        return {"saved": str(path)}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        live.clients.add(socket)
        await socket.send_text(json.dumps(live.state(), ensure_ascii=False))
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            live.clients.discard(socket)

    return app


def save_record(live: Live, result: str | None) -> Path:
    """Protocol log and advice history to data/raw/live/logs, and a row in my_battles / preview_records."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(live.started))
    record = {"started": stamp, "result": result, "team": live.roster, "preview": live.preview,
              "lineup": live.lineup, "log": live.session.log, "advice": live.history}
    path = LOG_DIR / f"{stamp}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    try:
        from pokesim.db.models import MyBattle, PreviewRecord
        from pokesim.db.session import Session

        with Session() as s:
            s.add(MyBattle(events=[{"kind": "record", **record}], result=result))
            if live.preview:
                s.add(PreviewRecord(source="obs", side_team=[to_id(x) for x in live.roster],
                                    opp_team=[to_id(x) for x in live.preview["species"]],
                                    brought=[to_id(x) for x in live.lineup or []], brought_complete=True,
                                    won=None if result is None else result == "win", rating=None))
            s.commit()
    except Exception as e:  # noqa: BLE001 - the JSON file is the primary record
        print("database record failed:", repr(e)[:200])
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=str(ROOT / "data/my_teams/baton.json"))
    parser.add_argument("--policy", help="fixed-team policy checkpoint (trained with this team order)")
    parser.add_argument("--no-lookahead", action="store_true")
    parser.add_argument("--no-screen", action="store_true", help="OBS: only the team preview, enter turns by hand")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--obs-source", help="OBS capture source name: watch it for the team preview")
    parser.add_argument("--obs-host", default="localhost")
    parser.add_argument("--obs-port", type=int, default=4455)
    parser.add_argument("--obs-password", default=os.environ.get("OBS_PASSWORD", ""), help="default: $OBS_PASSWORD")
    parser.add_argument("--obs-fps", type=float, default=4.0)
    parser.add_argument("--obs-record", type=Path, help="save changed frames here (e.g. data/raw/obs_frames)")
    args = parser.parse_args()
    live = Live(Path(args.team), args.policy, lookahead=not args.no_lookahead)
    live.read_screen = not args.no_screen
    obs = None
    if args.obs_source:
        obs = {"source": args.obs_source, "host": args.obs_host, "port": args.obs_port,
               "password": args.obs_password, "fps": args.obs_fps, "record": args.obs_record}
    uvicorn.run(create_app(live, obs), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
