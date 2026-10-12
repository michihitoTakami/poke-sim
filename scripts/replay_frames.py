"""Run the battle-screen reader over saved OBS frames and print the events it would send.

  uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/20261013 [--from T0 --to T1] [--apply]

Frames are the --obs-record output (named by capture time). The opponent's 6 come from the first team preview
in the range (or --opp). With --apply the events also go through BattleSession, as the server would do, and the
advice is printed at each decision.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from pokesim.live.screen import ScreenTracker, Vocab
from pokesim.vision.battle import ScreenReader
from pokesim.vision.obs_client import is_team_preview


def frames(paths: list[Path], t0: float | None, t1: float | None):
    for p in sorted(paths, key=lambda p: float(p.stem)):
        t = float(p.stem)
        if (t0 is None or t >= t0) and (t1 is None or t <= t1):
            yield t, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument("--team", default="data/my_teams/baton.json")
    ap.add_argument("--opp", help="comma-separated species (default: read the team preview)")
    ap.add_argument("--apply", action="store_true", help="feed BattleSession and print advice at decisions")
    ap.add_argument("--readings", action="store_true", help="print every frame's raw reading")
    args = ap.parse_args()

    sets = json.loads(Path(args.team).read_text())
    paths = [p for d in args.dirs for p in Path(d).glob("*.jpg")]
    reader, tracker, session, translator, advisor = ScreenReader(), None, None, None, None
    try:
        from pokesim.rl.advise import ja_names

        ja = ja_names()
    except Exception:  # noqa: BLE001 - no DB
        ja = {}
    opp = args.opp.split(",") if args.opp else None
    lineup_hint: list[str] = []
    t_read = []
    for t, p in frames(paths, args.t0, args.t1):
        img = cv2.imread(str(p))
        if tracker is None:
            if opp is None and is_team_preview(img):
                from pokesim.vision.preview import PreviewReader

                slots = PreviewReader().read(img)
                opp = [s.species or (s.candidates[0][0] if s.candidates else "") for s in slots]
                print(f"{p.stem} team preview: {opp}")
            if opp is None:
                continue
            tracker = ScreenTracker(Vocab.build(sets, opp, ja), maxhp=maxhp(sets))
        if is_team_preview(img):
            continue
        t1 = time.time()
        r = reader.read(img)
        t_read.append(time.time() - t1)
        if args.readings:
            print(f"   {p.stem} msg={r.message!r} opp={r.opp} me={r.me} cmd={r.decision}")
        for ev in tracker.feed(r):
            print(f"{p.stem} {ev}")
            if args.apply:
                session, translator, advisor = apply(ev, sets, opp, lineup_hint, session, translator, advisor)
    if t_read:
        print(f"read: mean {sum(t_read) / len(t_read):.3f}s  max {max(t_read):.3f}s over {len(t_read)} frames")


def maxhp(sets: list[dict]) -> dict[str, int]:
    from pokesim.live.session import BattleSession

    return {i["species"]: i["maxhp"] for i in BattleSession(sets).info}


def apply(ev, sets, opp, lineup, session, translator, advisor):
    from pokesim.live.advisor import Advisor
    from pokesim.live.events import EventTranslator
    from pokesim.live.session import BattleSession

    if session is None:
        session = BattleSession(sets)
        session.preview(opp)
        translator = EventTranslator(session)
        advisor = Advisor(sets, policy="checkpoints/ppo_baton_any.zip")
    if ev["type"] == "switch" and ev["side"] == "me" and ev["species"] not in lineup:
        lineup.append(ev["species"])
        rest = [s["species"] for s in sets if s["species"] not in lineup]
        order = [next(i for i, s in enumerate(sets) if s["species"] == sp) for sp in (lineup + rest)[:3]]
        if len(lineup) == 1:
            session.select(order)
    if ev["type"] in ("turn", "force_switch"):
        if ev["type"] == "turn":
            translator.apply([{"type": "turn"}])
        adv = advisor.advise(session.decision(force_switch=ev["type"] == "force_switch"))
        print("    advice:", [(label, round(p, 2)) for label, p in adv.actions[:3]], "win", adv.win_prob)
    else:
        translator.apply([ev])
    return session, translator, advisor


if __name__ == "__main__":
    main()
