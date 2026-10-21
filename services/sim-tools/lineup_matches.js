// Real three-on-three games from preview. Actors see their private request and the public six, never foe sets.
const {PublicObservation} = require('./public_observation');
const {PlanPolicy, choosePlannedPublic, VERSION} = require('./public_plans');
const {observedStep} = require('./public_policy');
const {health} = require('./public_observation');
const {chooseCounterplay, VERSION: COUNTERPLAY_VERSION} = require('./counterplay_policy');

// Diagnostics read only the actor's public observation and own request. They do not add
// reward: a payload matters to selection only when the real continuation can win with it.
function relayTrace() {
  const transfers = [];
  let pending = null, lastActive = null;
  return {transfers, observe(view) {
    const me = view.public.mons.p1[view.public.active.p1];
    if (!me) return;
    const active = view.public.active.p1;
    if (lastActive && active !== lastActive && pending) {
      if (Object.values(me.boosts).some(n => n > 0) || me.volatiles.includes('substitute'))
        transfers.push({from: lastActive, to: active, boosts: {...me.boosts},
          substitute: me.volatiles.includes('substitute'), hp: health(me.condition).fraction});
      pending = null;
    }
    if (me.lastMove === 'batonpass' && view.phase === 'switch' && health(me.condition).hp > 0) pending = active;
    lastActive = active;
  }};
}

function lineupMatches(req, {startBattle, step, extractChannelMessages}) {
  const cfg = req.lineupMatches;
  if (cfg.oursPolicy && !['planned', 'counterplay'].includes(cfg.oursPolicy)) throw Error('unknown lineup actor');
  const trials = [];
  const seeds = cfg.seeds || [0];
  if (!seeds.length || new Set(seeds).size !== seeds.length ||
      seeds.some(s => !Number.isInteger(s) || s < 0 || s > 65535)) throw Error('invalid lineup seeds');
  const actionValue = cfg.actionValueRun ? require('./action_value').loadActionValue(cfg.actionValueRun) : null;
  if (!cfg.ours?.length || !cfg.worlds?.length) throw Error('lineup matches require candidates and worlds');
  for (let candidate = 0; candidate < cfg.ours.length; candidate++) {
    for (let world = 0; world < cfg.worlds.length; world++) for (const seed of seeds) for (const opponent of cfg.policies || ['tactical', 'disrupt', 'expert']) {
      const sample = {p1: {team: cfg.ours[candidate]}, p2: {team: cfg.worlds[world]}};
      const battle = startBattle(sample, req.format, world + 1 + seed * 1000003);
      const row = {candidate, world, opponent, outcome: null, target: null, turns: 0, error: null};
      if (cfg.seeds) row.seed = seed;
      if (actionValue) row.actionValueChanges = 0;
      try {
        // Initial entry/weather events are real simulator events. Replace only the shown six, not their sets.
        const history = battle.log.filter(l => !l.startsWith('|poke|'));
        const at = history.findIndex(l => l.startsWith('|teampreview'));
        history.splice(at, 0, ...['p1', 'p2'].flatMap(side => cfg.preview[side].map(s => `|poke|${side}|${s}, L50|`)));
        const obs = new PublicObservation(history, battle.dex, extractChannelMessages, true);
        const plans = {p1: new PlanPolicy(), p2: new PlanPolicy()};
        let pos = battle.log.length, requests = 0;
        const relay = relayTrace(); row.relay = relay;
        const choose = side => {
          const view = obs.view(side);
          if (side.id === 'p1') relay.observe(view);
          const selection = (side.id === 'p1' ? (cfg.oursPolicy === 'counterplay' ? chooseCounterplay(view, battle.dex) :
            plans.p1.choose(view, battle.dex)) :
            opponent === 'counterplay' ? chooseCounterplay(view, battle.dex) :
              opponent === 'planned' ? plans.p2.choose(view, battle.dex) :
              choosePlannedPublic(view, opponent, battle.dex));
          const chosen = side.id === 'p1' && actionValue ? actionValue.choose(view, battle.dex, selection) : selection;
          if (side.id === 'p1' && actionValue && chosen[0] !== selection[0]) row.actionValueChanges++;
          return chosen[0];
        };
        while (!battle.ended && battle.turn <= (cfg.maxTurns ?? 160)) {
          if (++requests > 1400) throw Error('lineup request guard exhausted');
          const a = choose(battle.p1), b = choose(battle.p2);
          observedStep(battle, a, b, step, choose);
          obs.append(battle.log.slice(pos), true); pos = battle.log.length;
        }
        row.outcome = !battle.ended ? 'unfinished' : battle.winner === 'p1' ? 'win' : battle.winner === 'p2' ? 'loss' : 'draw';
        row.target = {win: 1, loss: 0, draw: .5}[row.outcome] ?? null;
        relay.observe(obs.view(battle.p1));
      } catch (e) { row.outcome = 'failed'; row.error = String(e.stack || e); }
      finally { row.turns = battle.turn; battle.destroy(); }
      trials.push(row);
    }
  }
  return {version: VERSION + '+' + COUNTERPLAY_VERSION + '+ours:' + (cfg.oursPolicy || 'planned') +
    (actionValue ? '+q:' + actionValue.sha : ''),
    informationScope: 'public_observation', trials};
}
module.exports = {lineupMatches};
