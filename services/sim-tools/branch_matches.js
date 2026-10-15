// Offline root-action comparisons. No HP score, deadline, policy selection or live-planner changes.
const {createHash} = require('crypto');

function evaluateBranches(req, {setUp, choices, step, policy, State}) {
  const cfg = req.branches;
  const seeds = cfg.seeds, names = cfg.continuations, maxTurns = cfg.maxTurns;
  if (!Array.isArray(seeds) || !seeds.length || seeds.some(s => !Number.isSafeInteger(s) || s < 0) ||
      new Set(seeds).size !== seeds.length) throw new Error('branches.seeds must be distinct nonnegative integers');
  if (!Number.isInteger(maxTurns) || maxTurns < 1) throw new Error('branches.maxTurns must be positive');
  if (!Array.isArray(names) || !names.length || new Set(names).size !== names.length) {
    throw new Error('branches.continuations must be distinct policy pairs');
  }
  const policies = names.map(name => {
    if (typeof name !== 'string' || name.split('/').length !== 2) throw new Error('expected p1/p2 policy pair');
    return name.split('/').map(policy);
  });
  if (!Array.isArray(req.samples) || !req.samples.length) throw new Error('no sampled worlds');
  const bases = req.samples.map(sample => {
    const b = setUp(sample, req.format, 1);
    if (b.ended || b.requestState !== 'move' || b.p1.activeRequest?.wait || b.p2.activeRequest?.wait) {
      throw new Error('branches supports normal turn-start decisions only');
    }
    return {snapshot: JSON.stringify(State.serializeBattle(b)), options: choices(b, b.p1)};
  });
  const actions = cfg.actions ?? [...new Set(bases.flatMap(b => b.options.map(x => x[1])))];
  if (!Array.isArray(actions) || !actions.length || actions.some(a => typeof a !== 'string') ||
      new Set(actions).size !== actions.length) throw new Error('branches.actions must be distinct labels');
  const trials = [];
  bases.forEach((base, world) => {
    for (const seed of seeds) {
      // The same initial chance stream for every action and continuation. Different actions can consume it
      // differently; these are paired trials, not a promise of identical damage/accuracy rolls thereafter.
      const bytes = createHash('sha256').update(`branch-matches:${seed}`).digest();
      const rng = [0, 2, 4, 6].map(offset => bytes.readUInt16LE(offset));
      names.forEach((name, k) => {
        const [p1Policy, p2Policy] = policies[k];
        for (const action of actions) {
          const battle = State.deserializeBattle(base.snapshot);
          battle.resetRNG(rng.slice());
          const start = battle.turn;
          const row = {world, seed, continuation: name, action, opponentAction: null,
                       outcome: null, target: null, turns: 0, error: null};
          try {
            const choice = base.options.find(([, label]) => label === action)?.[0];
            if (!choice) throw new Error(`root action unavailable in this world: ${action}`);
            // Fix the opponent's root decision before submitting ours. It does not see our candidate action.
            const opponent = p2Policy(battle, battle.p2);
            row.opponentAction = choices(battle, battle.p2).find(([c]) => c === opponent)?.[1] ?? opponent;
            step(battle, choice, opponent);
            let requests = 0;
            while (!battle.ended && battle.turn - start < maxTurns) {
              if (++requests > maxTurns * 8 + 16) throw new Error('continuation request guard exhausted');
              step(battle, p1Policy(battle, battle.p1), p2Policy(battle, battle.p2));
            }
            row.outcome = !battle.ended ? 'unfinished' :
              battle.winner === 'p1' ? 'win' : battle.winner === 'p2' ? 'loss' : 'draw';
            row.target = {win: 1, loss: 0, draw: 0.5}[row.outcome] ?? null;
          } catch (e) {
            row.outcome = 'failed';
            row.error = String(e.message || e);
          } finally {
            row.turns = battle.turn - start;
            battle.destroy();
          }
          trials.push(row);
        }
      });
    }
  });
  return {actions, trials};
}

module.exports = {evaluateBranches};
