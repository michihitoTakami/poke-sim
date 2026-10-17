// Real three-on-three games from preview. Actors see their private request and the public six, never foe sets.
const {PublicObservation} = require('./public_observation');
const {PlanPolicy, choosePlannedPublic, VERSION} = require('./public_plans');
const {observedStep} = require('./public_policy');

function lineupMatches(req, {startBattle, step, extractChannelMessages}) {
  const cfg = req.lineupMatches;
  const trials = [];
  if (!cfg.ours?.length || !cfg.worlds?.length) throw Error('lineup matches require candidates and worlds');
  for (let candidate = 0; candidate < cfg.ours.length; candidate++) {
    for (let world = 0; world < cfg.worlds.length; world++) for (const opponent of cfg.policies || ['tactical', 'disrupt', 'expert']) {
      const sample = {p1: {team: cfg.ours[candidate]}, p2: {team: cfg.worlds[world]}};
      const battle = startBattle(sample, req.format, world + 1);
      const row = {candidate, world, opponent, outcome: null, target: null, turns: 0, error: null};
      try {
        // Initial entry/weather events are real simulator events. Replace only the shown six, not their sets.
        const history = battle.log.filter(l => !l.startsWith('|poke|'));
        const at = history.findIndex(l => l.startsWith('|teampreview'));
        history.splice(at, 0, ...['p1', 'p2'].flatMap(side => cfg.preview[side].map(s => `|poke|${side}|${s}, L50|`)));
        const obs = new PublicObservation(history, battle.dex, extractChannelMessages, true);
        const plans = {p1: new PlanPolicy(), p2: new PlanPolicy()};
        let pos = battle.log.length, requests = 0;
        const choose = side => {
          const view = obs.view(side);
          return (side.id === 'p1' ? plans.p1.choose(view, battle.dex) :
            choosePlannedPublic(view, opponent, battle.dex))[0];
        };
        while (!battle.ended && battle.turn <= (cfg.maxTurns ?? 160)) {
          if (++requests > 1400) throw Error('lineup request guard exhausted');
          const a = choose(battle.p1), b = choose(battle.p2);
          observedStep(battle, a, b, step, choose);
          obs.append(battle.log.slice(pos), true); pos = battle.log.length;
        }
        row.outcome = !battle.ended ? 'unfinished' : battle.winner === 'p1' ? 'win' : battle.winner === 'p2' ? 'loss' : 'draw';
        row.target = {win: 1, loss: 0, draw: .5}[row.outcome] ?? null;
      } catch (e) { row.outcome = 'failed'; row.error = String(e.stack || e); }
      finally { row.turns = battle.turn; battle.destroy(); }
      trials.push(row);
    }
  }
  return {version: VERSION, informationScope: 'public_observation', trials};
}
module.exports = {lineupMatches};
