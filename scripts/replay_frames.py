"""Run the battle-screen reader over saved OBS frames and print the events it would send.

  uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/20261013 [--from T0 --to T1] [--apply]
  uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/* --records data/processed/replayed

Frames are the --obs-record output (named by capture time). Each team preview starts a battle (its opponent 6
are recognised there, or given with --opp). --apply feeds the events through BattleSession as the server does and
prints the advice at each decision; --records writes one record per battle (protocol log, lineup, result) in the
format of data/raw/live/logs, rebuilt with the current reader (for scripts/eval_records.py).
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from pokesim.live.events import EventTranslator
from pokesim.live.screen import ScreenTracker, Vocab
from pokesim.live.session import BattleSession
from pokesim.rl.myteam import to_id
from pokesim.vision.battle import ScreenReader
from pokesim.vision.obs_client import is_team_preview


def frames(paths: list[Path], t0: float | None, t1: float | None):
    for p in sorted(paths, key=lambda p: float(p.stem)):
        t = float(p.stem)
        if (t0 is None or t >= t0) and (t1 is None or t <= t1):
            yield t, p


class Game:
    """One battle: tracker + session (lineup registered from our switch-ins, as the server does)."""

    def __init__(self, sets: list[dict], opp: list[str], ja: dict, started: float, advisor=None):
        self.sets, self.opp, self.started, self.advisor = sets, opp, started, advisor
        self.session = BattleSession(sets)
        self.session.preview(opp)
        self.translator = EventTranslator(self.session)
        self.tracker = ScreenTracker(Vocab.build(sets, opp, ja), maxhp={i["species"]: i["maxhp"]
                                                                         for i in self.session.info})
        self.seen: list[str] = []
        self.result: str | None = None

    def lineup(self) -> list[str]:
        rest = [s["species"] for s in self.sets if s["species"] not in self.seen]
        return (self.seen + rest)[:3]

    def apply(self, ev: dict) -> None:
        if ev["type"] == "switch" and ev["side"] == "me" and ev["species"] not in self.seen:
            self.seen.append(ev["species"])
            self.session.select([next(i for i, s in enumerate(self.sets) if s["species"] == sp)
                                 for sp in self.lineup()])
        if ev["type"] == "win":
            self.result = "win" if ev["winner"] == "me" else "loss"
        if ev["type"] == "force_switch":
            self.advise(force_switch=True)
            return
        try:
            self.translator.apply([{k: v for k, v in ev.items() if k != "missed"}])
        except Exception as e:  # noqa: BLE001 - as the server: one bad read must not stop the battle
            print("    could not apply:", ev, repr(e)[:120])
            return
        if ev["type"] == "turn" and not ev.get("missed"):
            self.advise()

    def advise(self, force_switch: bool = False) -> None:
        if self.advisor is None:
            return
        adv = self.advisor.advise(self.session.decision(force_switch=force_switch), self.session.last_move(),
                                  self.session.log)
        print("    advice:", [(label, round(p, 2)) for label, p in adv.actions[:3]], "win", adv.win_prob,
              "opp:", [(label, round(p, 2)) for label, p in adv.opponent[:3]])

    def record(self) -> dict:
        return {"started": time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started)), "result": self.result,
                "team": [s["species"] for s in self.sets], "preview": {"species": self.opp},
                "lineup": self.lineup(), "log": self.session.log, "advice": [], "source": "replayed frames"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument("--team", default="data/my_teams/baton.json")
    ap.add_argument("--opp", help="comma-separated species (default: read the team preview)")
    ap.add_argument("--apply", action="store_true", help="feed BattleSession and print advice at decisions")
    ap.add_argument("--records", help="write one record per battle into this directory")
    ap.add_argument("--readings", action="store_true", help="print every frame's raw reading")
    args = ap.parse_args()

    sets = json.loads(Path(args.team).read_text())
    paths = [p for d in args.dirs for p in Path(d).glob("*.jpg")]
    try:
        from pokesim.rl.advise import ja_names

        ja = ja_names()
    except Exception:  # noqa: BLE001 - no DB
        ja = {}
    advisor = None
    if args.apply:
        from pokesim.live.advisor import Advisor

        advisor = Advisor(sets, policy="checkpoints/ppo_baton_any.zip")
    reader, preview_reader = ScreenReader(), None
    games: list[Game] = []
    game, preview_gone = None, True
    t_read = []
    for t, p in frames(paths, args.t0, args.t1):
        img = cv2.imread(str(p))
        if is_team_preview(img):
            if preview_gone:  # a new battle
                if args.opp:
                    opp = args.opp.split(",")
                else:
                    if preview_reader is None:
                        from pokesim.vision.preview import PreviewReader

                        preview_reader = PreviewReader()
                    slots = preview_reader.read(img)
                    opp = [s.species or (s.candidates[0][0] if s.candidates else "") for s in slots]
                print(f"{p.stem} team preview: {opp}")
                game = Game(sets, opp, ja, t, advisor)
                games.append(game)
            preview_gone = False
            continue
        preview_gone = True
        if game is None or game.tracker.finished:
            continue
        t1 = time.time()
        r = reader.read(img)
        t_read.append(time.time() - t1)
        if args.readings:
            print(f"   {p.stem} msg={r.message!r} opp={r.opp} me={r.me} cmd={r.decision}")
        for ev in game.tracker.feed(r):
            print(f"{p.stem} {ev}")
            if args.apply or args.records:
                game.apply(ev)
    if t_read:
        print(f"read: mean {sum(t_read) / len(t_read):.3f}s  max {max(t_read):.3f}s over {len(t_read)} frames")
    if args.records:
        out = Path(args.records)
        out.mkdir(parents=True, exist_ok=True)
        for g in games:
            if g.result is None and not any(line.startswith("|turn|") for line in g.session.log):
                continue
            rec = g.record()
            (out / f"{rec['started']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            print("record:", rec["started"], rec["result"], rec["lineup"], to_id(",".join(g.opp)))


if __name__ == "__main__":
    main()
