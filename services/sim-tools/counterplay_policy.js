// Stress opponent for relay teams. Inputs are the actor's private request and public protocol only.
const {legalChoices, health} = require('./public_observation');
const {chooseBatonPublic} = require('./public_policy');
const {estimateDamage, megaSource} = require('./public_damage');
const VERSION = 'public-counterplay-v3';

function chooseCounterplay(view, dex) {
  const opts = legalChoices(view), own = view.request.side.pokemon.find(m => m.active);
  const enemy = view.side === 'p1' ? 'p2' : 'p1';
  const foe = view.public.mons[enemy][view.public.active[enemy]];
  const me = view.public.mons[view.side][view.public.active[view.side]];
  if (view.phase !== 'move' || !foe || !own || !me) return chooseBatonPublic(view, 'expert', dex);
  const move = mid => opts.find(([, label]) => label === `move:${mid}:mega`) ||
    opts.find(([, label]) => label === `move:${mid}`);
  const spec = m => dex.species.get(m.details?.split(',')[0] || m.species);
  const types = foe.types || spec(foe).types;
  const source = megaSource(view, dex, own), h = health(own.condition);
  const ability = foe.ability || (new Set(Object.values(spec(foe).abilities)).size === 1 ?
    String(spec(foe).abilities['0']).toLowerCase().replace(/[^a-z0-9]/g, '') : '');
  const canTarget = mid => {
    const m = dex.moves.get(mid);
    const bypass = ['moldbreaker', 'teravolt', 'turboblaze'].includes(source.ability) && foe.item !== 'abilityshield';
    if (!bypass && (ability === 'goodasgold' || (ability === 'magicbounce' && m.flags.reflectable))) return false;
    if (foe.volatiles.includes('substitute') && !m.flags.bypasssub && own.ability !== 'infiltrator') return false;
    if (!bypass && ability === 'soundproof' && m.flags.sound) return false;
    return !(own.ability === 'prankster' && types.includes('Dark'));
  };
  const hasBoost = Object.values(foe.boosts).some(n => n > 0);
  const statusPlay = !foe.moves.length || foe.moves.some(mid => {
    const m = dex.moves.get(mid); return m.category === 'Status' && (m.boosts || m.heal || m.selfSwitch ||
      ['protect', 'substitute'].includes(mid));
  });
  const best = chooseBatonPublic(view, 'greedy', dex), attack = dex.moves.get(best[1].split(':')[1]);
  const damage = best[1].startsWith('move:') && attack.category !== 'Status' ?
    estimateDamage(view, dex, best[1].endsWith(':mega') ? source : own, foe, attack) : 0;
  const full = health(foe.condition).fraction === 1;
  const protectedKO = full && (['focussash'].includes(foe.item) || ['sturdy', 'disguise'].includes(ability));
  const finish = !protectedKO && damage * .85 / .925 >=
    health(foe.condition).fraction * (spec(foe).baseStats.hp + 107);
  if (finish) return best;
  if (hasBoost && move('haze')) return move('haze');

  // Ghost Curse and Encore/phazing bypass Substitute in the pinned engine. Preserve their real flags.
  const ghost = (source.types || me.types || spec(source).types).includes('Ghost');
  const hasBench = view.request.side.pokemon.some(m => !m.active && health(m.condition).hp > 0);
  if (ghost && move('curse') && canTarget('curse') && !foe.volatiles.includes('curse') &&
      (hasBoost || statusPlay) && (h.hp > h.maxhp / 2 || hasBench)) return move('curse');
  const last = dex.moves.get(foe.lastMove);
  if (move('encore') && canTarget('encore') && !foe.volatiles.includes('encore') && last.exists &&
      last.category === 'Status' && !last.flags.failencore) return move('encore');
  if (statusPlay && move('taunt') && canTarget('taunt') && !foe.volatiles.includes('taunt')) return move('taunt');

  const threats = foe.moves.map(mid => dex.moves.get(mid)).filter(m => m.category !== 'Status');
  if (foe.moves.length < 4) for (const type of spec(foe).types) threats.push({type, basePower: 90,
    category: spec(foe).baseStats.atk >= spec(foe).baseStats.spa ? 'Physical' : 'Special'});
  const incoming = boosts => Math.max(0, ...threats.map(m => estimateDamage(view, dex, foe, source,
    {...m, accuracy: true}, foe.boosts, boosts) / .925));
  const hazards = view.public.conditions[enemy] || {};
  const rocks = Object.keys(hazards).some(k => k.toLowerCase().replace(/[^a-z]/g, '').includes('stealthrock'));
  if (move('stealthrock') && !rocks && statusPlay && incoming(me.boosts) < h.hp - h.maxhp / 8)
    return move('stealthrock');
  const canPhaze = (view.public.totalFainted?.[enemy] || 0) <
    Math.min(3, view.public.preview[enemy].length || 3) - 1;
  if (canPhaze && (hasBoost || (rocks && statusPlay))) {
    for (const mid of ['whirlwind', 'roar']) if (move(mid) && canTarget(mid) &&
        !foe.volatiles.includes('ingrain') && ability !== 'suctioncups') return move(mid);
  }

  if (canTarget('toxic')) {
    if (move('toxic') && !health(foe.condition).status && !types.some(t => ['Steel', 'Poison'].includes(t)))
      return move('toxic');
    if (move('soak') && !(types.length === 1 && types[0] === 'Water') &&
        ((move('toxic') && !health(foe.condition).status) || own.moves.some(mid =>
          ['Electric', 'Grass'].includes(dex.moves.get(mid).type) && dex.moves.get(mid).category !== 'Status')))
      return move('soak');
  }
  // Actually play DD/Shell Smash as well as special setup. Assess the resulting damage and lowered bulk,
  // rather than giving an intrinsic reward to boosts. A named stress rollout never consults hidden foe sets.
  if (statusPlay && !['tox', 'slp', 'frz'].includes(h.status) &&
      !me.volatiles.some(v => ['taunt', 'yawn', 'encore', 'curse'].includes(v))) {
    const setups = ['shellsmash', 'dragondance', 'swordsdance', 'quiverdance', 'nastyplot', 'calmmind', 'bulkup', 'irondefense'];
    for (const mid of setups) {
      if (!move(mid)) continue;
      const boosts = dex.moves.get(mid).boosts || {}, projected = {...me.boosts};
      for (const [stat, n] of Object.entries(boosts)) projected[stat] = Math.max(-6, Math.min(6, (projected[stat] || 0) + n));
      const nextDamage = Math.max(0, ...own.moves.map(m => estimateDamage(view, dex, source, foe,
        dex.moves.get(m), projected, foe.boosts)));
      const useful = nextDamage >= Math.max(1, damage * 1.2) &&
        Object.entries(boosts).some(([s, n]) => n > 0 && (me.boosts[s] || 0) < (s === 'spe' ? 3 : 4));
      if (useful && Math.max(incoming(me.boosts), incoming(projected)) < h.hp * .6) return move(mid);
    }
  }
  if (best[1].startsWith('move:') && damage > 0) return best;
  return chooseBatonPublic(view, 'expert', dex);
}
module.exports = {chooseCounterplay, VERSION};
