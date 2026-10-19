// Multi-turn options using only an actor's request and the public protocol prefix.
const {health, id, legalChoices} = require('./public_observation');
const {chooseBatonPublic} = require('./public_policy');
const {estimateDamage} = require('./public_damage');
const VERSION = 'public-plans-v4';
const stage = n => n >= 0 ? (2 + n) / 2 : 2 / (2 - n);

function assess(view, dex) {
  const side = view.side, enemy = side === 'p1' ? 'p2' : 'p1';
  const me = view.request.side.pokemon.find(m => m.active);
  const foe = view.public.mons[enemy][view.public.active[enemy]];
  if (!me || !foe) return null;
  const species = m => dex.species.get(m.details?.split(',')[0] || m.species);
  const mine = view.public.mons[side][view.public.active[side]] || {boosts: {}, volatiles: []};
  const stats = (m, s, boosts = {}) => {
    let value = (m.stats?.[s] ?? Math.floor((2 * species(m).baseStats[s] + 47) / 2) + 5) * stage(boosts[s] || 0);
    if (s !== 'spe') return value;
    if (health(m.condition).status === 'par' && m.ability !== 'quickfeet') value *= .5;
    if (m.item === 'choicescarf') value *= 1.5;
    const team = m === foe ? enemy : side;
    if (Object.keys(view.public.conditions?.[team] || {}).some(c => /tailwind/i.test(c))) value *= 2;
    const weather = id(view.public.weather);
    if ((weather === 'raindance' && m.ability === 'swiftswim') ||
        (weather === 'sunnyday' && m.ability === 'chlorophyll') ||
        (weather === 'sandstorm' && m.ability === 'sandrush') ||
        (['snow', 'snowscape', 'hail'].includes(weather) && m.ability === 'slushrush')) value *= 2;
    return value;
  };
  const estimate = (target, move, projected = {}) => {
    return estimateDamage(view, dex, foe, target, {...move, accuracy: true}, undefined, projected) / .925;
  };
  const attacks = foe.moves.map(m => dex.moves.get(m)).filter(m => m.category !== 'Status');
  if (foe.moves.length < 4) for (const type of species(foe).types) attacks.push({type, basePower: 90,
    category: species(foe).baseStats.atk >= species(foe).baseStats.spa ? 'Physical' : 'Special'});
  const hit = (mon, boosts) => Math.max(0, ...attacks.map(m => estimate(mon, m, boosts)));
  const priority = Math.max(0, ...attacks.filter(m => m.priority > 0).map(m => estimate(me, m, mine.boosts)));
  const current = health(me.condition), incoming = hit(me, mine.boosts);
  const critical = foe.volatiles.includes('focusenergy') || attacks.some(m => m.willCrit);
  const sheltered = mine.volatiles.includes('substitute') && foe.ability !== 'infiltrator' &&
    !attacks.some(m => m.flags?.sound || m.flags?.bypasssub || m.multihit);
  const bench = view.request.side.pokemon.filter(m => !m.active && health(m.condition).hp > 0);
  // Required receiver speed is computed before deciding to protect again or hand off. Re-evaluated every turn.
  const receivers = bench.filter(m => m.moves.some(mid => dex.moves.get(mid).category !== 'Status'));
  const trick = (view.history || []).reduce((on, l) => /^\|-fieldstart\|(?:move: )?Trick Room/.test(l) ? true :
    /^\|-fieldend\|(?:move: )?Trick Room/.test(l) ? false : on, false);
  const faster = (m, boosts) => trick ? stats(m, 'spe', boosts) < stats(foe, 'spe', foe.boosts) :
    stats(m, 'spe', boosts) > stats(foe, 'spe', foe.boosts);
  return {me, foe, mine, current, incoming, priority, critical, sheltered, receivers, hit, faster,
    canAct: priority < current.hp && (faster(me, mine.boosts) || incoming < current.hp || sheltered),
    signature: JSON.stringify([id(me.details.split(',')[0]), view.public.active[enemy], foe.moves.slice().sort(),
      foe.ability, foe.item, critical])};
}

class PlanPolicy {
  constructor() { this.plan = null; this.events = []; }
  choose(view, dex) {
    const fallback = () => chooseBatonPublic(view, 'expert', dex);
    const opts = legalChoices(view), a = assess(view, dex);
    if (!a || view.phase !== 'move') return fallback();
    const {me, foe, mine, current, incoming, critical, sheltered, receivers, canAct} = a;
    const move = mid => opts.find(([, l]) => l === `move:${mid}:mega`) || opts.find(([, l]) => l === `move:${mid}`);
    const attack = () => {
      // The public tactical policy already compares finishing attacks and safe switches.
      const filtered = {...view, request: {...view.request, active: view.request.active.map(x => ({...x,
        moves: x.moves.map(m => ({...m, disabled: m.disabled || dex.moves.get(m.id).category === 'Status'}))}))}};
      return legalChoices(filtered).length ? chooseBatonPublic(filtered, 'tactical', dex) : fallback();
    };
    const end = reason => { this.events.push({turn: view.turn, plan: this.plan?.kind, event: reason}); this.plan = null; };
    if (this.plan && this.plan.signature !== a.signature) end('new-public-information');
    if (this.plan?.kind === 'relay' && !move('batonpass')) end('baton-unavailable');
    if (this.plan && view.turn - this.plan.start > 4) end('deadline');
    const strategy = require('./relay_strategy').relayStrategy(view, dex);
    if (strategy) {
      if (this.plan?.kind !== strategy.goal) {
        this.plan = {kind: strategy.goal, signature: a.signature, start: view.turn};
        this.events.push({turn: view.turn, plan: strategy.goal, event: 'start'});
      }
      return strategy.choice;
    }
    if (this.plan && !['relay', 'protected-setup'].includes(this.plan.kind)) end('goal-reconsidered');
    const disrupted = foe.moves.some(m => ['roar', 'whirlwind', 'haze', 'encore', 'taunt'].includes(m));
    if (this.plan?.kind === 'protected-setup' && (!sheltered || mine.volatiles.includes('yawn') || critical || disrupted))
      end('protection-lost-or-critical-threat');
    if (!this.plan && !mine.volatiles.includes('yawn')) {
      if (move('batonpass') && receivers.length) this.plan = {kind: 'relay', signature: a.signature, start: view.turn};
      else if (!critical && !disrupted && move('calmmind') && (sheltered || (move('substitute') &&
          current.hp > current.maxhp / 2 && incoming < current.maxhp / 4 &&
          mine.substituteBrokenTurn !== view.turn - 1 &&
          (foe.moves.includes('yawn') || dex.moves.get(foe.lastMove).heal))))
        this.plan = {kind: 'protected-setup', signature: a.signature, start: view.turn};
      if (this.plan) this.events.push({turn: view.turn, plan: this.plan.kind, event: 'start'});
    }
    if (this.plan?.kind === 'protected-setup') {
      if (!sheltered) return move('substitute');
      // Goal: one safe boost, then cash in the shield. Never keep accumulating boosts without attacking.
      if ((mine.boosts.spa || 0) < 1 && incoming < current.hp && move('calmmind')) return move('calmmind');
      end('goal-reached');
      return attack();
    }
    if (this.plan?.kind === 'relay') {
      const payload = Object.values(mine.boosts).some(n => n > 0) || sheltered;
      const residual = ['tox', 'psn', 'brn'].includes(current.status) ? current.maxhp / 8 : 0;
      const ready = receivers.some(m => a.faster(m, mine.boosts) && a.hit(m, mine.boosts) < health(m.condition).hp);
      const preparation = fallback();
      // The expert controller can safely alternate Protect/Substitute for a slow receiver even when
      // a direct hit is large. Do not replace that protected turn with a premature handoff.
      if (['move:protect', 'move:detect', 'move:substitute'].includes(preparation[1])) return preparation;
      // Deadline includes priority and residuals. A failed speed race cannot justify losing the whole payload.
      const deadline = 2 * incoming + residual >= current.hp || a.priority + residual >= current.hp ||
        mine.volatiles.includes('yawn') || critical || foe.moves.some(m => ['haze', 'roar', 'whirlwind', 'taunt'].includes(m));
      if (payload && canAct && (ready || deadline) && preparation[1] === 'move:batonpass') {
        end(ready ? 'receiver-ready' : 'handoff-deadline'); return move('batonpass');
      }
      if (critical && move('calmmind')) {
        // Rebuild the choice without special-defence-only preparation; critical hits bypass those boosts.
        end('critical-plan-reconsidered');
        return attack();
      }
    }
    return fallback();
  }
}
const choosePlannedPublic = (view, name, dex) => name === 'planned' ? new PlanPolicy().choose(view, dex) :
  chooseBatonPublic(view, name, dex);
module.exports = {PlanPolicy, choosePlannedPublic, assess, VERSION};
