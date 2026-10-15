#!/bin/bash
# Showdown ladder collection until stopped: batches of 20 battles, restarted after each batch (and whenever
# scripts/ladder.py exits to reload changed code / priors). Records: data/raw/ladder/. See docs/operations.md.
#   start:  cd /data/poke-sim/collect && PLANNER=on setsid nohup <repo>/scripts/collect.sh > /dev/null 2>&1 < /dev/null &
#   stop:   touch $COLLECT_DIR/STOP   (finishes the current batch first)
#   status: tail $COLLECT_DIR/status.log ; tail -f $COLLECT_DIR/ladder.log
# Environment: PLANNER (off | shadow | on, default on), COLLECT_DIR (default /data/poke-sim/collect),
#   POLICIES (default: the two baton policies, alternating per battle), EXTRA (more ladder.py options, e.g. --belief)
cd "$(dirname "$0")/.." || exit 1
C=${COLLECT_DIR:-/data/poke-sim/collect}
POLICIES=${POLICIES:-"checkpoints/ppo_baton_v2.zip checkpoints/ppo_baton_items.zip"}
mkdir -p "$C"
exec 9>"$C/collect.lock"
if ! flock -n 9; then
  echo "collection is already running for $C" >&2
  exit 1
fi
rm -f "$C/STOP"
echo "start $(date) pid $$ planner ${PLANNER:-on} extra '${EXTRA:-}' commit $(git rev-parse --short HEAD)" >> "$C/status.log"
i=0; fails=0
while [ ! -e "$C/STOP" ]; do
  i=$((i+1)); t0=$(date +%s)
  before=$(ls data/raw/ladder/*.json 2>/dev/null | wc -l)
  # shellcheck disable=SC2086
  if [ -n "${PYTHON:-}" ]; then
    "$PYTHON" scripts/ladder.py --battles 20 --planner "${PLANNER:-on}" --policy $POLICIES ${EXTRA:-} \
        >> "$C/ladder.log" 2>&1
  else
    uv run python scripts/ladder.py --battles 20 --planner "${PLANNER:-on}" --policy $POLICIES ${EXTRA:-} \
        >> "$C/ladder.log" 2>&1
  fi
  code=$?
  after=$(ls data/raw/ladder/*.json 2>/dev/null | wc -l)
  echo "batch $i exit $code, +$((after - before)) battles (total $after), $(( $(date +%s) - t0 ))s, $(date)" >> "$C/status.log"
  if [ "$code" -eq 78 ]; then
    touch "$C/STOP"
    echo "search quality gate failed; stopped for inspection $(date)" >> "$C/status.log"
    break
  fi
  if [ $((after - before)) -eq 0 ]; then  # nothing saved (login / network trouble): back off, up to 30 min
    fails=$((fails+1)); sleep $(( fails < 6 ? 60 * 2 ** (fails - 1) : 1800 ))
  else
    fails=0; sleep 5
  fi
done
echo "stopped $(date)" >> "$C/status.log"
