// Paired, one-turn counterplay over ALL legal root replies, independent of continuation policies.
// These are sampled chance paths, never a certificate or a calibrated probability.
const {createHash} = require('crypto');
const {PublicObservation} = require('./public_observation');
const {PlanPolicy} = require('./public_plans');
const {observedStep} = require('./public_policy');
const VERSION = 'paired-root-counterplay-v1';

function rootTactics(req, deps) {
  const cfg = req.rootTactics, cells = {}, kinds = {};
  const deadline = Date.now() + (cfg.deadlineMs ?? 1200);
  const actions = cfg.actions, seeds = [0, 1, 2];
  if (!actions?.length || new Set(actions).size !== actions.length) throw Error('distinct root actions required');
  for (const [world, sample] of req.samples.entries()) {
    const base = deps.setUp(sample, req.format, 1);
    let snapshot, ours, replies;
    try {
      if (base.requestState !== 'move' || base.p1.activeRequest?.wait || base.p2.activeRequest?.wait)
        throw Error('root counterplay requires a normal move request');
      snapshot = JSON.stringify(deps.State.serializeBattle(base));
      ours = deps.choices(base, base.p1); replies = deps.choices(base, base.p2);
      for (const label of actions) {
        const move = label.startsWith('move:') ? base.dex.moves.get(label.split(':')[1]) : null;
        kinds[label] = move?.category === 'Status' ? 'status' : move ? 'attack' : 'switch';
      }
    } finally { base.destroy(); }
    for (const [reply, replyLabel] of replies) {
      const key = `${world}:${replyLabel}`;
      cells[key] = {};
      for (const label of actions) {
        const choice = ours.find(([, l]) => l === label)?.[0];
        if (!choice) throw Error(`root action unavailable: ${label}`);
        const values = [];
        for (const seed of seeds) {
          if (Date.now() >= deadline) return {version: VERSION, complete: false, reason: 'deadline'};
          const b = deps.State.deserializeBattle(snapshot);
          const bytes = createHash('sha256').update(`root-counterplay:${seed}`).digest();
          b.resetRNG([0, 2, 4, 6].map(i => bytes.readUInt16LE(i)));
          try {
            const before = [b.p1, b.p2].map(s => ({alive: s.pokemon.filter(m => m.hp > 0).length,
              hp: s.pokemon.reduce((n, m) => n + m.hp / m.maxhp, 0)}));
            const obs = new PublicObservation(req.observationLog, b.dex, deps.extractChannelMessages,
              b.dex.currentMod.startsWith('champions'));
            for (const side of b.sides) {
              obs.bindActive(side.id, side.active[0]?.name);
              obs.bindActive(side.id, side.active[0]?.illusion?.name);
            }
            const plans = {p1: new PlanPolicy(), p2: new PlanPolicy()};
            let at = b.log.length;
            const choose = side => plans[side.id].choose(obs.view(side), b.dex)[0];
            observedStep(b, choice, reply, deps.step, choose, ['p2']);
            obs.append(b.log.slice(at), true); at = b.log.length;
            // Finish pending pivot/Baton Pass choices and the rest of this turn, without a second move turn.
            for (let guard = 0; !b.ended && b.requestState === 'switch'; guard++) {
              if (guard >= 8) throw Error('root switch guard exhausted');
              const a = b.p1.activeRequest && !b.p1.activeRequest.wait ? choose(b.p1) : 'pass';
              const c = b.p2.activeRequest && !b.p2.activeRequest.wait ? choose(b.p2) : 'pass';
              observedStep(b, a, c, deps.step, choose);
              obs.append(b.log.slice(at), true); at = b.log.length;
            }
            const lost = before[0].alive - b.p1.pokemon.filter(m => m.hp > 0).length;
            const ko = before[1].alive - b.p2.pokemon.filter(m => m.hp > 0).length;
            values.push({lost, ko, damage: before[1].hp - b.p2.pokemon.reduce((n, m) => n + m.hp / m.maxhp, 0)});
          } finally { b.destroy(); }
        }
        cells[key][label] = Object.fromEntries(['lost', 'ko', 'damage'].map(k =>
          [k, values.reduce((n, v) => n + v[k], 0) / seeds.length]));
      }
    }
  }
  return {version: VERSION, complete: true, actions, kinds, cells, seeds,
    scope: 'all legal root replies in sampled sets; three paired seeds; not exhaustive chance or hidden sets'};
}
module.exports = {rootTactics, VERSION};
