// Offline root-action comparisons. No HP score, deadline, policy selection or live-planner changes.
const {createHash} = require('crypto');
const {PublicObservation, legalChoices} = require('./public_observation');
const {choosePublic, VERSION} = require('./public_policy');

function evaluateBranches(req, {setUp, choices, step, policy, State, extractChannelMessages}) {
  const cfg = req.branches;
  const scope = cfg.informationScope || 'sampled_full_information_continuations';
  if (!['sampled_full_information_continuations', 'public_observation'].includes(scope)) {
    throw new Error('unknown branch information scope');
  }
  const isPublic = scope === 'public_observation';
  const seeds = cfg.seeds, names = cfg.continuations, maxTurns = cfg.maxTurns;
  if (!Array.isArray(seeds) || !seeds.length || seeds.some(s => !Number.isSafeInteger(s) || s < 0) ||
      new Set(seeds).size !== seeds.length) throw new Error('branches.seeds must be distinct nonnegative integers');
  if (!Number.isInteger(maxTurns) || maxTurns < 1) throw new Error('branches.maxTurns must be positive');
  if (!Array.isArray(names) || !names.length || new Set(names).size !== names.length) {
    throw new Error('branches.continuations must be distinct policy pairs');
  }
  const policies = names.map(name => {
    if (typeof name !== 'string' || name.split('/').length !== 2) throw new Error('expected p1/p2 policy pair');
    if (isPublic) {
      for (const p of name.split('/')) if (!['greedy', 'tactical', 'expert', 'disrupt'].includes(p)) {
        throw new Error(`unknown public policy ${p}`);
      }
      return name.split('/');
    }
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
          if (isPublic) row.trajectory = [];
          try {
            const observer = isPublic ? new PublicObservation(cfg.observationLog, battle.dex,
              extractChannelMessages, battle.dex.currentMod.startsWith('champions')) : null;
            let logPosition = battle.log.length, decisionStep = 0;
            const decide = (side, policyName, fixedAction = null) => {
              if (side.activeRequest?.wait || !side.activeRequest) return 'pass';
              const view = observer.view(side);
              const options = legalChoices(view);
              const selection = fixedAction === null ? choosePublic(view, policyName, battle.dex) :
                options.find(([, label]) => label === fixedAction);
              if (!selection) throw new Error(`root action unavailable in public request: ${fixedAction}`);
              row.trajectory.push({step: decisionStep, side: side.id, phase: view.phase,
                observation: view, legal: options.map(([, label]) => label), action: selection[1],
                behavior: policyName, rootIntervention: decisionStep === 0, target: null});
              return selection[0];
            };
            const advance = (a, b) => {
              step(battle, a, b);
              if (observer) observer.append(battle.log.slice(logPosition), true);
              logPosition = battle.log.length;
              decisionStep++;
            };
            const choice = base.options.find(([, label]) => label === action)?.[0];
            if (!choice) throw new Error(`root action unavailable in this world: ${action}`);
            // Fix the opponent's root decision before submitting ours. It does not see our candidate action.
            const opponent = isPublic ? decide(battle.p2, p2Policy) : p2Policy(battle, battle.p2);
            row.opponentAction = choices(battle, battle.p2).find(([c]) => c === opponent)?.[1] ?? opponent;
            advance(isPublic ? decide(battle.p1, p1Policy, action) : choice, opponent);
            let requests = 0;
            while (!battle.ended && battle.turn - start < maxTurns) {
              if (++requests > maxTurns * 8 + 16) throw new Error('continuation request guard exhausted');
              const ours = isPublic ? decide(battle.p1, p1Policy) : p1Policy(battle, battle.p1);
              const theirs = isPublic ? decide(battle.p2, p2Policy) : p2Policy(battle, battle.p2);
              advance(ours, theirs);
            }
            row.outcome = !battle.ended ? 'unfinished' :
              battle.winner === 'p1' ? 'win' : battle.winner === 'p2' ? 'loss' : 'draw';
            row.target = {win: 1, loss: 0, draw: 0.5}[row.outcome] ?? null;
          } catch (e) {
            row.outcome = 'failed';
            row.error = String(e.message || e);
          } finally {
            if (isPublic) for (const decision of row.trajectory) {
              decision.target = row.target === null ? null : decision.side === 'p1' ? row.target : 1 - row.target;
            }
            row.turns = battle.turn - start;
            battle.destroy();
          }
          trials.push(row);
        }
      });
    }
  });
  return {actions, trials, ...(isPublic ? {informationScope: scope, policyVersion: VERSION} : {})};
}

module.exports = {evaluateBranches};
