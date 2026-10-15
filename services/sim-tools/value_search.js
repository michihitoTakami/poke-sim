// Matrix rollouts under the same observation-constrained reference used to train V_ref.
// Nonterminal leaves cross the Python boundary as actor-visible JSON; no HP score enters learned payoffs.
const {createHash} = require('crypto');
const {PublicObservation} = require('./public_observation');
const {choosePublic, observedStep, VERSION} = require('./public_policy');

function evaluateValue(req, {setUp, choices, step, score, State, extractChannelMessages}) {
  const start = Date.now(), names = req.continuations || ['expert/tactical', 'expert/disrupt'];
  const seeds = req.seeds ?? 1, rollout = req.rollout ?? 2, seedBase = req.seedBase ?? 0;
  if (!Number.isInteger(seeds) || seeds < 1 || !Number.isInteger(rollout) || rollout < 0 ||
      !Number.isSafeInteger(seedBase) || seedBase < 0) throw new Error('invalid value search budget');
  if (!names.length || names.some(n => !['expert/tactical', 'expert/disrupt'].includes(n))) {
    throw new Error('value search requires the frozen public reference and opponent pool');
  }
  const states = req.samples.map(sample => {
    const b = setUp(sample, req.format, 1);
    const st = {snapshot: JSON.stringify(State.serializeBattle(b)), entryLog: b.pokesimEntryLog,
      p1: b.ended ? [['pass', 'pass']] : choices(b, b.p1),
      p2: b.ended ? [['pass', 'pass']] : choices(b, b.p2), leaves: [], errors: [], done: 0};
    st.invalid = st.p1.map(() => st.p2.map(() => 0));
    b.destroy();
    return st;
  });
  for (let k = 0; k < seeds; k++) {
    if (k > 0 && req.deadlineMs && Date.now() - start > req.deadlineMs) break;
    for (const name of names) for (const st of states) {
      const [ours, theirs] = name.split('/');
      for (let i = 0; i < st.p1.length; i++) for (let j = 0; j < st.p2.length; j++) {
        const b = State.deserializeBattle(st.snapshot), seed = seedBase + k;
        const bytes = createHash('sha256').update(`branch-matches:${seed}`).digest();
        b.resetRNG([0, 2, 4, 6].map(n => bytes.readUInt16LE(n)));
        try {
          const observer = new PublicObservation(req.observationLog, b.dex, extractChannelMessages,
            b.dex.currentMod.startsWith('champions'));
          observer.append(st.entryLog || []);
          for (const side of b.sides) {
            observer.bindActive(side.id, side.active[0]?.name);
            observer.bindActive(side.id, side.active[0]?.illusion?.name);
          }
          let at = b.log.length;
          let root = true;
          const advance = (a, c) => {
            observedStep(b, a, c, step, side => choose(side, side.id === 'p1' ? ours : theirs),
              root ? [] : ['p1', 'p2']);
            root = false;
            observer.append(b.log.slice(at), true); at = b.log.length;
          };
          const choose = (side, policy) => !side.activeRequest || side.activeRequest.wait ? 'pass' :
            choosePublic(observer.view(side), policy, b.dex)[0];
          const rootTurn = b.turn;
          if (!b.ended) advance(st.p1[i][0], st.p2[j][0]);
          let guard = 0;
          while (!b.ended && (b.turn - rootTurn < 1 + rollout || b.requestState !== 'move')) {
            if (++guard > (rollout + 1) * 12 + 24) throw new Error('value continuation guard exhausted');
            // Both choices use the same pre-action public history.
            const a = choose(b.p1, ours), c = choose(b.p2, theirs);
            advance(a, c);
          }
          const terminal = b.ended ? (b.winner === 'p1' ? 1 : b.winner === 'p2' ? 0 : .5) : null;
          const observation = terminal === null ? observer.view(b.p1) : null;
          if (observation && observation.phase !== 'move') throw new Error('unsupported value leaf phase');
          st.leaves.push({i, j, seed, continuation: name, terminal, observation,
            hpDiagnostic: score(b, 1, 0, 'terminal', req.lostPathPenalty || 0, 0)});
        } catch (e) {
          st.invalid[i][j]++;
          if (st.errors.length < 3) st.errors.push(String(e.message || e));
        } finally { b.destroy(); }
      }
      st.done++;
    }
  }
  return {results: states.map(st => ({p1: st.p1.map(x => x[1]), p2: st.p2.map(x => x[1]),
    leaves: st.leaves, invalid: st.invalid, value_errors: st.errors, passes: [st.done, seeds * names.length]})),
    policyVersion: VERSION, elapsedMs: Date.now() - start};
}

module.exports = {evaluateValue};
