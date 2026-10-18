// Executable relay goals from public observations. No species whitelist, hidden sets, or raw boost reward.
const {health, id, legalChoices} = require('./public_observation');
const {estimateDamage, megaSource} = require('./public_damage');
const VERSION = 'public-relay-goals-v1';
const stage = n => n >= 0 ? (2 + n) / 2 : 2 / (2 - n);
const DISRUPTION = new Set(['curse', 'encore', 'taunt', 'haze', 'clearsmog', 'roar', 'whirlwind']);

function sunlight(view, dex) {
  if (id(view.public.weather) !== 'sunnyday') return null;
  let elapsed = 0;
  for (let i = view.history.length - 1; i >= 0; i--) {
    const p = view.history[i].split('|');
    if (p[1] !== '-weather') continue;
    if (id(p[2]) !== 'sunnyday') return null;
    if (p.includes('[upkeep]')) { elapsed++; continue; }
    const source = p.find(x => x.startsWith('[of] '))?.slice(5);
    let item = null, species = null;
    for (let j = i - 1; source && j >= 0; j--) {
      const q = view.history[j].split('|');
      if (q[2] !== source) continue;
      if (q[1] === '-mega' && item === null) item = id(q[4]);
      if (q[1] === '-item' && item === null) item = id(q[3]);
      if (q[1] === '-enditem' && item === null) item = '';
      if (['switch', 'drag', 'detailschange', '-formechange'].includes(q[1])) { species = q[3].split(',')[0]; break; }
    }
    const stone = species && dex.species.get(species).requiredItem;
    if (item === null && stone) item = id(stone);
    const low = item === 'heatrock' ? 8 : 5, high = item === null ? 8 : low;
    return {min: Math.max(0, low - elapsed), max: Math.max(0, high - elapsed), elapsed};
  }
  return null;
}

function relayStrategy(view, dex) {
  if (view.phase !== 'move') return null;
  const options = legalChoices(view), own = view.request.side.pokemon;
  const me = own.find(m => m.active), side = view.side, enemy = side === 'p1' ? 'p2' : 'p1';
  const foe = view.public.mons[enemy][view.public.active[enemy]];
  const mine = view.public.mons[side][view.public.active[side]];
  if (!me || !foe || !mine) return null;
  const species = m => dex.species.get(m.details?.split(',')[0] || m.species);
  const move = mid => (view.request.active?.[0]?.moves || []).some(m => m.id === mid && m.pp > 0) && (options.find(([, label]) => label === `move:${mid}:mega`) ||
    options.find(([, label]) => label === `move:${mid}`));
  const answer = (choice, goal, evidence = {}) => choice ? {choice, goal, version: VERSION, evidence} : null;
  const current = health(me.condition), payload = mine.boosts;
  const projectedMe = megaSource(view, dex, me);
  const bench = own.filter(m => !m.active && health(m.condition).hp > 0);
  const trickRoom = view.history.reduce((on, line) => /^\|-fieldstart\|(?:move: )?Trick Room/.test(line) ? true :
    /^\|-fieldend\|(?:move: )?Trick Room/.test(line) ? false : on, false);
  const speed = (mon, boosts = {}, upper = false) => {
    let n = mon.stats?.spe ?? Math.floor((species(mon).baseStats.spe + 52) * (upper ? 1.1 : 1));
    n *= stage(boosts.spe || 0);
    if (health(mon.condition).status === 'par' && mon.ability !== 'quickfeet') n *= .5;
    if (mon.item === 'choicescarf') n *= 1.5;
    const team = mon === foe ? enemy : side;
    if (Object.keys(view.public.conditions[team] || {}).some(k => /tailwind/i.test(k))) n *= 2;
    const weather = id(view.public.weather);
    if ((weather === 'sunnyday' && mon.ability === 'chlorophyll') ||
        (weather === 'raindance' && mon.ability === 'swiftswim') ||
        (weather === 'sandstorm' && mon.ability === 'sandrush') ||
        (['snow', 'snowscape', 'hail'].includes(weather) && mon.ability === 'slushrush')) n *= 2;
    return n;
  };
  const faster = (mon, boosts = {}) => !trickRoom && speed(mon, boosts) > speed(foe, foe.boosts, true);
  const attacks = foe.moves.map(mid => dex.moves.get(mid)).filter(m => m.category !== 'Status');
  // No generic attack is invented for the dedicated physical plan: scout before committing to defence.
  if (!attacks.length && !mine.volatiles.includes('curse')) return null;
  if (foe.moves.length < 4) for (const type of species(foe).types) attacks.push({type, basePower: 90,
    category: species(foe).baseStats.atk >= species(foe).baseStats.spa ? 'Physical' : 'Special'});
  // A public safety margin, not an inferred set: bound offensive EVs/nature and known Life Orb.
  const upperFoe = {...foe, stats: Object.fromEntries(['atk', 'spa', 'def', 'spd'].map(s =>
    [s, Math.floor((species(foe).baseStats[s] + 52) * 1.1)]))};
  const hit = (target, boosts = {}, publicView = view, list = attacks) => Math.max(0, ...list.map(m =>
    estimateDamage(publicView, dex, upperFoe, target, {...m, accuracy: true}, foe.boosts, boosts) / .925 *
      (foe.item === 'lifeorb' ? 1.3 : 1)));
  const priorityHit = (target, boosts = {}) => hit(target, boosts, view, attacks.filter(m => m.priority > 0));
  const residual = (mon, curse = false) => {
    const h = health(mon.condition), status = h.status;
    if (mon.ability === 'magicguard') return 0;
    let n = status === 'brn' ? h.maxhp / 16 : status === 'psn' ? h.maxhp / 8 :
      status === 'tox' ? h.maxhp / 4 : 0;
    if (id(view.public.weather) === 'sandstorm' && !species(mon).types.some(t => ['Rock', 'Steel', 'Ground'].includes(t)))
      n += h.maxhp / 16;
    return n + (curse ? h.maxhp / 4 : 0);
  };
  const healing = mon => {
    if (!mon.moves.some(mid => ['recover', 'roost', 'slackoff', 'synthesis', 'moonlight', 'morningsun'].includes(mid))) return 0;
    const weather = id(view.public.weather);
    const sunHeal = mon.moves.some(mid => ['synthesis', 'moonlight', 'morningsun'].includes(mid));
    return health(mon.condition).maxhp * (sunHeal ? weather === 'sunnyday' ? 2 / 3 :
      ['raindance', 'sandstorm', 'snowscape', 'hail'].includes(weather) ? .25 : .5 : .5);
  };
  const endpoint = mon => mon.moves.some(mid => estimateDamage(view, dex, mon, foe, dex.moves.get(mid)) > 0);
  const entryCost = mon => {
    if (mon.ability === 'magicguard' || mon.item === 'heavydutyboots') return 0;
    const conditions = view.public.conditions[side] || {}, types = species(mon).types;
    const rocks = Object.keys(conditions).some(k => /stealthrock/i.test(k)) ?
      health(mon.condition).maxhp / 8 * 2 ** dex.getEffectiveness('Rock', types) : 0;
    const layers = Object.entries(conditions).find(([k]) => /spikes/i.test(k) && !/toxic/i.test(k))?.[1] || 0;
    return rocks + (!types.includes('Flying') && mon.ability !== 'levitate' ?
      health(mon.condition).maxhp * [0, 1 / 8, 1 / 6, 1 / 4][Math.min(3, layers)] : 0);
  };
  const recoveryHit = (mon, boosts) => {
    const types = species(mon).types.filter(t => t !== 'Flying');
    return Math.max(hit(mon, boosts), mon.moves.includes('roost') ?
      hit({...mon, types: types.length ? types : ['Normal']}, boosts) : 0);
  };
  const ready = (mon, boosts) => {
    const h = health(mon.condition), damage = Math.max(hit(mon, boosts), recoveryHit(mon, boosts)), r = residual(mon);
    return !['slp', 'frz', 'tox'].includes(h.status) && endpoint(mon) && faster(mon, boosts) && 2 * (damage + r) < h.hp - entryCost(mon) && healing(mon) > damage + r;
  };
  const protectedNow = mine.volatiles.includes('substitute') && foe.ability !== 'infiltrator' &&
    !attacks.some(m => m.flags?.sound || m.flags?.bypasssub || m.multihit);
  const canPass = () => priorityHit(me, payload) + residual(me) < current.hp &&
    (faster(me, payload) || protectedNow || hit(me, payload) + residual(me) < current.hp);
  const disrupted = foe.moves.some(mid => DISRUPTION.has(mid)) || mine.volatiles.includes('yawn');

  // Leave a plausible finishing attack to the tactical controller and the root win certificate.
  // Public HP is a percentage; use maximum possible HP/defence rather than treating it as exact HP.
  const foeHp = health(foe.condition).fraction * (species(foe).baseStats.hp + 107);
  const finishing = !foe.volatiles.includes('substitute') &&
    !(health(foe.condition).fraction === 1 && (['focussash'].includes(foe.item) ||
      ['sturdy', 'disguise'].includes(foe.ability) || species(foe).abilities['0'] === 'Disguise')) &&
    (view.request.active?.[0]?.moves || []).some(m => !m.disabled && m.pp > 0 &&
      estimateDamage(view, dex, projectedMe, upperFoe, {...dex.moves.get(m.id), accuracy: true},
        payload, foe.boosts) * .85 / .925 >= foeHp);
  if (finishing) return null;

  if (mine.volatiles.includes('curse')) {
    const switches = options.filter(([, label]) => label.startsWith('switch:')).map(choice => {
      const mon = own[Number(choice[0].split(' ')[1]) - 1], h = health(mon.condition);
      return {choice, value: (h.hp - entryCost(mon) - hit(mon) - residual(mon)) / h.maxhp + (endpoint(mon) ? .1 : 0)};
    }).sort((a, b) => b.value - a.value);
    if (switches.length) return answer(switches[0].choice, 'clear-curse-with-normal-switch');
    // Trapping may leave no escape. Do not prescribe an unavailable switch or relabel a last attack as a loss.
    return null;
  }

  if (['slp', 'frz', 'tox'].includes(current.status)) return null;

  const physical = hit(projectedMe, payload, view, attacks.filter(m => m.category === 'Physical'));
  const special = hit(projectedMe, payload, view, attacks.filter(m => m.category === 'Special'));
  const pass = move('batonpass'), protect = move('protect') || move('detect'), sub = move('substitute');
  const passivePasser = me.ability === 'speedboost' && pass && me.moves.every(mid => dex.moves.get(mid).category === 'Status');
  if (passivePasser && !disrupted && !trickRoom) {
    const recipients = bench.filter(endpoint);
    if (physical > 0 && physical >= special) {
      const receiver = recipients.find(mon => ready(mon, payload));
      if (receiver && (payload.def || 0) > 0 && canPass())
        return answer(pass, 'receiver-defence-ready', {receiver: receiver.details, defence: payload.def});
      const iron = move('irondefense'), projected = {...payload, def: Math.min(6, (payload.def || 0) + 2)};
      const nextHit = Math.max(hit(me, projected), priorityHit(me, payload));
      if (iron && projected.def > (payload.def || 0) && faster(me, payload) &&
          current.hp - nextHit - residual(me) > priorityHit(me, projected) + residual(me) &&
          recipients.some(mon => ready(mon, {...projected, spe: Math.min(6, (payload.spe || 0) + 1)})))
        return answer(iron, 'prepare-receiver-defence', {defence: projected.def});
    }
    const sun = sunlight(view, dex);
    const fire = attacks.some(m => m.category === 'Special' && (m.type === 'Fire' || m.id === 'weatherball'));
    const clear = {...view, public: {...view.public, weather: ''}};
    const usefulSunWait = recipients.some(mon => healing(mon) > 0 &&
      hit(mon, payload) - hit(mon, payload, clear) > health(mon.condition).maxhp * .1);
    if (sun?.max > 0 && special > physical && fire && usefulSunWait &&
        !foe.moves.some(mid => dex.moves.get(mid).boosts) && canPass()) {
      if (protectedNow && recipients.some(mon => faster(mon, payload) &&
          hit(mon, {...payload, spd: Math.min(6, (payload.spd || 0) + 1)}) < healing(mon)))
        return answer(pass, 'pass-intact-shield', {sun});
      const lastProtect = ['protect', 'detect'].includes(mine.lastMove);
      if (protect && !lastProtect && current.hp > residual(me))
        return answer(protect, 'wait-out-sun', {sun});
      if (sub && !protectedNow && current.hp - Math.floor(current.maxhp / 4) >
          priorityHit(me, payload) + residual(me) && foe.ability !== 'infiltrator' &&
          !attacks.some(m => m.flags?.sound || m.flags?.bypasssub || m.multihit))
        return answer(sub, 'wait-out-sun', {sun});
    }
  }

  // With a usable recovery move, first reach the receiver's survival threshold, then cash in or pass.
  const heal = ['moonlight', 'synthesis', 'morningsun', 'recover', 'slackoff', 'roost'].map(move).find(Boolean);
  const calm = move('calmmind'), incoming = hit(projectedMe, payload), r = residual(me);
  if (calm && heal && (payload.spe || 0) > 0 && special > physical && faster(projectedMe, payload) && !disrupted &&
      !foe.volatiles.includes('focusenergy') && !attacks.some(m => m.willCrit)) {
    const h = healing(me), healHit = recoveryHit(projectedMe, payload);
    const afterHeal = Math.min(current.maxhp, current.hp + h) - healHit - r;
    if (current.hp < current.maxhp && current.hp <= 2 * (healHit + r) &&
        (afterHeal > current.hp + 1 || current.hp <= healHit + r) && afterHeal > 0)
      return answer(heal, 'recover-before-preparing', {healing: h, incoming});
    if (pass && (payload.spa || 0) >= 1 && (payload.spd || 0) >= 1 && canPass()) {
      const receiver = bench.find(mon => ready(mon, payload));
      if (receiver) return answer(pass, 'special-receiver-ready', {receiver: receiver.details});
    }
    const projected = {...payload, spa: Math.min(6, (payload.spa || 0) + 1), spd: Math.min(6, (payload.spd || 0) + 1)};
    const nextHit = hit(projectedMe, projected);
    if ((payload.spd || 0) < 2 && priorityHit(me, payload) < current.hp &&
        current.hp - nextHit - r > current.maxhp * .1 && nextHit + r < h)
      return answer(calm, 'prepare-special-survival', {incoming: nextHit, specialDefence: projected.spd});
  }

  // An Unaware defender can use burn against a revealed physical setup without mirroring its boosts.
  const burn = move('willowisp'), foeAbility = foe.ability ||
    (Object.keys(species(foe).abilities).length === 1 ? id(species(foe).abilities['0']) : '');
  if (me.ability === 'unaware' && burn && physical > special &&
      ((foe.boosts.atk || 0) > 0 || foe.moves.some(mid => dex.moves.get(mid).boosts?.atk > 0)) &&
      !foe.volatiles.includes('substitute') && !health(foe.condition).status &&
      !species(foe).types.includes('Fire') &&
      !['guts', 'thermalexchange', 'waterveil', 'waterbubble', 'purifyingsalt', 'magicbounce', 'goodasgold'].includes(foeAbility) &&
      priorityHit(me, payload) < current.hp && (faster(me, payload) || incoming < current.hp))
    return answer(burn, 'deny-physical-setup');
  return null;
}

module.exports = {relayStrategy, sunlight, VERSION};
