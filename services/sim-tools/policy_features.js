// Numeric features of an actor's view and of each of its legal choices, for the learned public policies.
// The same vectors are produced by src/pokesim/learn/features.py from the same view JSON (parity-tested), so
// the net trained in Python runs here on exactly the inputs it saw. Only client-visible JSON is read: the actor's
// own request, the public state and history. Opponent stats are the neutral estimate the hand-written policy uses.
const {health, id} = require('./public_observation');

const VERSION = 'policy-features-v1';
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const TYPES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground',
  'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'];
const STATUSES = ['brn', 'par', 'psn', 'tox', 'slp', 'frz', 'fnt'];
const VOLATILES = ['substitute', 'yawn', 'taunt', 'encore', 'disable', 'leechseed', 'confusion', 'saltcure',
  'perishsong', 'curse', 'ingrain', 'aquaring'];
const ITEMS = ['leftovers', 'sitrusberry', 'mentalherb', 'airballoon', 'focussash', 'choicescarf', 'choiceband',
  'choicespecs', 'eviolite', 'lumberry', 'rockyhelmet', 'heavydutyboots'];
const ABILITIES = ['unaware', 'goodasgold', 'speedboost', 'levitate', 'regenerator', 'illusion', 'moldbreaker',
  'waterabsorb', 'stormdrain', 'flashfire', 'sturdy', 'magicbounce', 'intimidate', 'poisonheal'];
const CONDITIONS = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'reflect', 'lightscreen', 'auroraveil',
  'tailwind'];
const WEATHERS = ['sunnyday', 'raindance', 'sandstorm', 'snow', 'hail'];
const TERRAINS = ['electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain'];
const PROTECTS = ['protect', 'detect', 'kingsshield', 'spikyshield', 'banefulbunker', 'obstruct', 'silktrap',
  'burningbulwark'];
const HAZES = ['haze', 'clearsmog'];
const stage = s => s >= 0 ? (2 + s) / 2 : 2 / (2 - s);
const effectId = s => id(String(s || '').replace(/^(?:move|ability|item): /, ''));
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// Dex access limited to what the exported training dex also holds.
function speciesOf(dex, name) {
  const s = dex.species.get(name);
  return s && s.exists ? s : null;
}

function statOf(dex, mon, s, boosts, own) {
  const spec = speciesOf(dex, own ? mon.details.split(',')[0] : mon.species);
  const base = spec ? spec.baseStats[s] : 80;
  let value = own && mon.stats && mon.stats[s] !== undefined ? mon.stats[s] : Math.floor((2 * base + 31 + 16) / 2) + 5;
  value *= stage((boosts || {})[s] || 0);
  if (s === 'spe' && health(mon.condition).status === 'par') value *= .5;
  if (s === 'spe' && id(mon.item) === 'choicescarf') value *= 1.5;
  return value;
}

function maxHpOf(dex, mon, own) {
  if (own) return health(mon.condition).maxhp;
  const spec = speciesOf(dex, mon.species);
  return Math.floor((2 * (spec ? spec.baseStats.hp : 80) + 31 + 16) / 2) + 60;
}

function hpOf(dex, mon, own) {
  return own ? health(mon.condition).hp : health(mon.condition).fraction * maxHpOf(dex, mon, own);
}

function typesOf(dex, mon, own) {
  const spec = speciesOf(dex, own ? mon.details.split(',')[0] : mon.species);
  return spec ? spec.types : [];
}

function effectiveness(dex, type, types) {
  if (!dex.getImmunity(type, types)) return null;  // immune
  return dex.getEffectiveness(type, types);
}

// Expected damage of one move: the hand-written policy's estimate (public_policy.js), accuracy-weighted.
function damage(dex, source, target, move, sourceBoosts, targetBoosts, sourceOwn, targetOwn) {
  if (!move || move.category === 'Status') return 0;
  const types = typesOf(dex, target, targetOwn);
  const eff = effectiveness(dex, move.type, types);
  if (eff === null) return 0;
  const ability = id(target.ability || ''), item = id(target.item || '');
  if ((move.type === 'Ground' && (ability === 'levitate' || item === 'airballoon')) ||
      (move.type === 'Water' && ['waterabsorb', 'stormdrain'].includes(ability)) ||
      (move.type === 'Fire' && ability === 'flashfire')) return 0;
  const physical = move.category === 'Physical';
  let power = move.basePower || (move.id === 'bodypress' ? 80 : 60);
  if (Array.isArray(move.multihit)) power *= 3; else power *= move.multihit || 1;
  if (id(target.ability || '') === 'unaware') sourceBoosts = {};
  if (id(source.ability || '') === 'unaware') targetBoosts = {};
  let result = ((22 * power * statOf(dex, source, physical ? 'atk' : 'spa', sourceBoosts, sourceOwn) /
    Math.max(1, statOf(dex, target, physical ? 'def' : 'spd', targetBoosts, targetOwn))) / 50 + 2) * .925;
  if (typesOf(dex, source, sourceOwn).includes(move.type)) result *= 1.5;
  result *= 2 ** eff;
  if (physical && health(source.condition).status === 'brn') result *= .5;
  if (typeof move.accuracy === 'number') result *= move.accuracy / 100;
  return result;
}

function trickRoom(history) {
  let on = false;
  for (const line of history || []) {
    if (/^\|-fieldstart\|(?:move: )?Trick Room/.test(line)) on = true;
    if (/^\|-fieldend\|(?:move: )?Trick Room/.test(line)) on = false;
  }
  return on;
}

// A context shared by the state and action features of one view.
function context(view, dex) {
  const side = view.side, foeSide = side === 'p1' ? 'p2' : 'p1';
  const own = view.request.side.pokemon;
  const me = own.find(m => m.active) || own[0];
  const pub = view.public;
  const myPublic = pub.mons[side][pub.active[side]] || {boosts: {}, volatiles: [], moves: []};
  const foe = pub.mons[foeSide][pub.active[foeSide]] || null;
  const knownThreats = foe ? foe.moves.map(m => dex.moves.get(m)).filter(m => m.exists && m.category !== 'Status') : [];
  const foeSpec = foe ? speciesOf(dex, foe.species) : null;
  const generic = foeSpec ? foeSpec.types.map(type => ({type, basePower: 90, accuracy: 100,
    category: foeSpec.baseStats.atk >= foeSpec.baseStats.spa ? 'Physical' : 'Special'})) : [];
  const threats = [...knownThreats, ...(!foe || foe.moves.length < 4 ? generic : [])];
  const incoming = target => foe ? Math.max(0, ...threats.map(m =>
    damage(dex, foe, target, m, foe.boosts, myPublic.boosts, false, true))) : 0;
  const bestDamage = source => foe ? Math.max(0, ...source.moves.map(mid =>
    damage(dex, source, foe, dex.moves.get(mid), source === me ? myPublic.boosts : {}, foe.boosts, true, false))) : 0;
  const room = trickRoom(view.history);
  const mySpeed = m => statOf(dex, m, 'spe', m === me ? myPublic.boosts : {}, true);
  const foeSpeed = foe ? statOf(dex, foe, 'spe', foe.boosts, false) : 0;
  const faster = m => room ? mySpeed(m) < foeSpeed : mySpeed(m) > foeSpeed;
  return {side, foeSide, own, me, pub, myPublic, foe, threats, incoming, bestDamage, room, mySpeed, foeSpeed, faster};
}

function speciesFeatures(out, prefix, spec) {
  for (const s of STATS) out[`${prefix}.base_${s}`] = (spec ? spec.baseStats[s] : 0) / 255;
  for (const t of TYPES) out[`${prefix}.type_${t}`] = spec && spec.types.includes(t) ? 1 : 0;
  out[`${prefix}.mega`] = spec && spec.isMega ? 1 : 0;
}

function itemFeatures(out, prefix, item, dex) {
  const i = id(item || '');
  for (const name of ITEMS) out[`${prefix}.item_${name}`] = i === name ? 1 : 0;
  const data = dex.items.get(i);
  out[`${prefix}.mega_item`] = data && data.exists && data.megaStone ? 1 : 0;
  out[`${prefix}.berry`] = data && data.exists && data.isBerry ? 1 : 0;
}

function abilityFeatures(out, prefix, ability) {
  const a = id(ability || '');
  for (const name of ABILITIES) out[`${prefix}.ability_${name}`] = a === name ? 1 : 0;
}

function stateFeatures(view, dex) {
  const c = context(view, dex), out = {};
  const {me, foe, myPublic, pub, own} = c;
  out.turn = Math.min(view.turn, 100) / 100;
  out.phase_switch = view.phase === 'switch' ? 1 : 0;
  out.trick_room = c.room ? 1 : 0;
  // actor's active
  const myHealth = health(me.condition);
  speciesFeatures(out, 'me', speciesOf(dex, me.details.split(',')[0]));
  out['me.hp'] = myHealth.fraction;
  for (const s of STATUSES) out[`me.status_${s}`] = myHealth.status === s ? 1 : 0;
  for (const b of BOOSTS) out[`me.boost_${b}`] = ((myPublic.boosts || {})[b] || 0) / 6;
  for (const v of VOLATILES) out[`me.volatile_${v}`] = (myPublic.volatiles || []).includes(v) ? 1 : 0;
  itemFeatures(out, 'me', me.item, dex);
  abilityFeatures(out, 'me', me.ability || me.baseAbility);
  const active = view.request.active && view.request.active[0] || {};
  out['me.can_mega'] = active.canMegaEvo ? 1 : 0;
  out['me.trapped'] = active.trapped ? 1 : 0;
  const slots = active.moves || [];
  out['me.pp_min'] = slots.length ? Math.min(...slots.map(m => (m.pp || 0) / Math.max(1, m.maxpp || 1))) : 0;
  out['me.moves'] = (me.moves || []).length / 4;
  out['me.faster'] = foe && c.faster(me) ? 1 : 0;
  out['me.speed_share'] = foe ? c.mySpeed(me) / (c.mySpeed(me) + c.foeSpeed) : .5;
  out['me.incoming'] = clip(c.incoming(me) / Math.max(1, hpOf(dex, me, true)), 0, 2);
  out['me.outgoing'] = foe ? clip(c.bestDamage(me) / Math.max(1, hpOf(dex, foe, false)), 0, 2) : 0;
  out['me.boosted'] = Object.values(myPublic.boosts || {}).some(v => v > 0) ? 1 : 0;
  // actor's bench: two slots in request order
  const bench = own.filter(m => m !== me);
  for (let k = 0; k < 2; k++) {
    const m = bench[k], prefix = `bench${k}`;
    const h = m ? health(m.condition) : {fraction: 0, status: 'fnt', hp: 0};
    out[`${prefix}.present`] = m ? 1 : 0;
    out[`${prefix}.alive`] = m && h.hp > 0 ? 1 : 0;
    out[`${prefix}.hp`] = m ? h.fraction : 0;
    for (const s of STATUSES) out[`${prefix}.status_${s}`] = m && h.status === s ? 1 : 0;
    speciesFeatures(out, prefix, m ? speciesOf(dex, m.details.split(',')[0]) : null);
    out[`${prefix}.mega_item`] = 0;
    if (m) { const data = dex.items.get(id(m.item || '')); out[`${prefix}.mega_item`] = data && data.exists && data.megaStone ? 1 : 0; }
    out[`${prefix}.faster`] = m && foe && h.hp > 0 && c.faster(m) ? 1 : 0;
    out[`${prefix}.incoming`] = m && foe && h.hp > 0 ? clip(c.incoming(m) / Math.max(1, h.hp), 0, 2) : 0;
    out[`${prefix}.outgoing`] = m && foe && h.hp > 0 ? clip(c.bestDamage(m) / Math.max(1, hpOf(dex, foe, false)), 0, 2) : 0;
  }
  // foe's active (public)
  const fh = foe ? health(foe.condition) : {fraction: 0, status: ''};
  out['foe.present'] = foe ? 1 : 0;
  speciesFeatures(out, 'foe', foe ? speciesOf(dex, foe.species) : null);
  out['foe.hp'] = foe ? fh.fraction : 0;
  for (const s of STATUSES) out[`foe.status_${s}`] = foe && fh.status === s ? 1 : 0;
  for (const b of BOOSTS) out[`foe.boost_${b}`] = foe ? ((foe.boosts || {})[b] || 0) / 6 : 0;
  for (const v of VOLATILES) out[`foe.volatile_${v}`] = foe && (foe.volatiles || []).includes(v) ? 1 : 0;
  const foeMoves = foe ? foe.moves.map(m => dex.moves.get(m)).filter(m => m.exists) : [];
  out['foe.moves_seen'] = foeMoves.length / 4;
  for (const cat of ['Physical', 'Special', 'Status']) out[`foe.moves_${cat}`] = foeMoves.filter(m => m.category === cat).length / 4;
  out['foe.max_power'] = Math.max(0, ...foeMoves.map(m => m.basePower || 0)) / 250;
  out['foe.priority'] = Math.max(0, ...foeMoves.map(m => m.priority || 0)) / 5;
  out['foe.item_known'] = foe && foe.item !== undefined ? 1 : 0;
  itemFeatures(out, 'foe', foe ? foe.item : '', dex);
  out['foe.ability_known'] = foe && foe.ability !== undefined ? 1 : 0;
  abilityFeatures(out, 'foe', foe ? foe.ability : '');
  const last = foe && foe.lastMove ? dex.moves.get(foe.lastMove) : null;
  out['foe.last_known'] = last && last.exists ? 1 : 0;
  for (const cat of ['Physical', 'Special', 'Status']) out[`foe.last_${cat}`] = last && last.exists && last.category === cat ? 1 : 0;
  out['foe.boosted'] = foe && Object.values(foe.boosts || {}).some(v => v > 0) ? 1 : 0;
  out['foe.incoming_hint'] = foe ? clip(c.incoming(me) / Math.max(1, hpOf(dex, me, true)), 0, 2) : 0;
  // foe's bench (public) and preview
  const foeMons = Object.entries(pub.mons[c.foeSide]).filter(([key]) => key !== pub.active[c.foeSide]).map(([, m]) => m);
  out['foe.bench_seen'] = foeMons.length / 3;
  out['foe.bench_alive'] = foeMons.filter(m => health(m.condition).hp > 0).length / 3;
  out['foe.bench_hp'] = foeMons.reduce((s, m) => s + health(m.condition).fraction, 0) / 3;
  for (const s of STATS) out[`foe.bench_base_${s}`] = foeMons.reduce((sum, m) => { const sp = speciesOf(dex, m.species); return sum + (sp ? sp.baseStats[s] : 0); }, 0) / (255 * 3);
  for (const t of TYPES) out[`foe.bench_type_${t}`] = foeMons.some(m => { const sp = speciesOf(dex, m.species); return sp && sp.types.includes(t); }) ? 1 : 0;
  const preview = pub.preview[c.foeSide].map(d => speciesOf(dex, d.split(',')[0]));
  out['foe.unseen'] = Math.max(0, preview.length - Object.keys(pub.mons[c.foeSide]).length) / 6;
  for (const s of STATS) out[`foe.preview_base_${s}`] = preview.reduce((sum, sp) => sum + (sp ? sp.baseStats[s] : 0), 0) / (255 * 6);
  for (const t of TYPES) out[`foe.preview_type_${t}`] = preview.filter(sp => sp && sp.types.includes(t)).length / 6;
  // field
  const weather = effectId(pub.weather), terrain = effectId(pub.terrain);
  for (const w of WEATHERS) out[`weather_${w}`] = weather === w ? 1 : 0;
  for (const t of TERRAINS) out[`terrain_${t}`] = terrain === t ? 1 : 0;
  for (const [label, role] of [['own', c.side], ['foe', c.foeSide]]) {
    const conditions = {};
    for (const [k, v] of Object.entries(pub.conditions[role] || {})) conditions[effectId(k)] = v;
    for (const cond of CONDITIONS) out[`${label}.condition_${cond}`] = (conditions[cond] || 0) / (cond === 'spikes' ? 3 : cond === 'toxicspikes' ? 2 : 1);
  }
  return out;
}

function actionFeatures(view, dex, choice, label) {
  const c = context(view, dex), out = {};
  const {me, foe, myPublic, own} = c;
  const parts = label.split(':');
  const kind = parts[0], mega = parts[2] === 'mega';
  out.kind_move = kind === 'move' ? 1 : 0;
  out.kind_mega = mega ? 1 : 0;
  out.kind_switch = kind === 'switch' ? 1 : 0;
  const move = kind === 'move' ? dex.moves.get(parts[1]) : null;
  const known = move && move.exists;
  for (const t of TYPES) out[`move.type_${t}`] = known && move.type === t ? 1 : 0;
  for (const cat of ['Physical', 'Special', 'Status']) out[`move.${cat}`] = known && move.category === cat ? 1 : 0;
  out['move.power'] = known ? (move.basePower || 0) / 200 : 0;
  out['move.accuracy'] = known ? (typeof move.accuracy === 'number' ? move.accuracy / 100 : 1) : 0;
  out['move.priority'] = known ? (move.priority || 0) / 5 : 0;
  const slot = known ? ((view.request.active && view.request.active[0] || {}).moves || []).find(m => m.id === move.id) : null;
  out['move.pp'] = slot ? (slot.pp || 0) / Math.max(1, slot.maxpp || 1) : 0;
  out['move.heal'] = known && ((move.flags && move.flags.heal) || move.drain) ? 1 : 0;
  out['move.protect'] = known && PROTECTS.includes(move.id) ? 1 : 0;
  out['move.substitute'] = known && move.id === 'substitute' ? 1 : 0;
  out['move.batonpass'] = known && move.id === 'batonpass' ? 1 : 0;
  out['move.pivot'] = known && move.selfSwitch && move.id !== 'batonpass' ? 1 : 0;
  out['move.phaze'] = known && move.forceSwitch ? 1 : 0;
  out['move.haze'] = known && HAZES.includes(move.id) ? 1 : 0;
  const boosts = known ? (move.target === 'self' && move.boosts ? move.boosts : (move.self && move.self.boosts) || {}) : {};
  out['move.setup'] = Object.values(boosts).some(v => v > 0) ? 1 : 0;
  for (const b of BOOSTS.slice(0, 5)) out[`move.boost_${b}`] = Math.max(0, boosts[b] || 0) / 2;
  out['move.status'] = known && move.status ? 1 : 0;
  out['move.sound'] = known && move.flags && move.flags.sound ? 1 : 0;
  out['move.contact'] = known && move.flags && move.flags.contact ? 1 : 0;
  out['move.multihit'] = known && move.multihit ? 1 : 0;
  out['move.recoil'] = known && move.recoil ? 1 : 0;
  out['move.last'] = known && myPublic.lastMove === move.id ? 1 : 0;
  out['move.stab'] = known && typesOf(dex, me, true).includes(move.type) ? 1 : 0;
  let eff = null;
  if (known && foe && move.category !== 'Status') eff = effectiveness(dex, move.type, typesOf(dex, foe, false));
  out['move.immune'] = known && foe && move.category !== 'Status' && eff === null ? 1 : 0;
  out['move.effectiveness'] = eff === null ? 0 : eff / 2;
  const dmg = known && foe ? damage(dex, me, foe, move, myPublic.boosts, foe.boosts, true, false) : 0;
  const foeHp = foe ? Math.max(1, hpOf(dex, foe, false)) : 1;
  out['move.damage'] = clip(dmg / foeHp, 0, 2);
  out['move.kills'] = foe && dmg >= hpOf(dex, foe, false) ? 1 : 0;
  // switch target
  const target = kind === 'switch' ? own[Number(choice.split(' ')[1]) - 1] : null;
  const th = target ? health(target.condition) : {fraction: 0, status: '', hp: 0};
  speciesFeatures(out, 'switch', target ? speciesOf(dex, target.details.split(',')[0]) : null);
  out['switch.hp'] = target ? th.fraction : 0;
  for (const s of STATUSES) out[`switch.status_${s}`] = target && th.status === s ? 1 : 0;
  out['switch.incoming'] = target && foe ? clip(c.incoming(target) / Math.max(1, th.hp), 0, 2) : 0;
  out['switch.outgoing'] = target && foe ? clip(c.bestDamage(target) / foeHp, 0, 2) : 0;
  out['switch.faster'] = target && foe && c.faster(target) ? 1 : 0;
  out['switch.mega_item'] = 0;
  if (target) { const data = dex.items.get(id(target.item || '')); out['switch.mega_item'] = data && data.exists && data.megaStone ? 1 : 0; }
  return out;
}

function vector(features, names) {
  const out = new Float32Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const v = features[names[i]];
    if (v === undefined) throw Error(`missing feature ${names[i]}`);
    out[i] = v;
  }
  return out;
}

module.exports = {VERSION, stateFeatures, actionFeatures, vector, damage, context};
