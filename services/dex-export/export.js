// Export the Showdown Champions dex (legal entries only) to JSON for the Python ingest.
// Usage: node services/dex-export/export.js <showdown_dir> <format_id> <out.json>
const path = require('path');
const fs = require('fs');

const [showdownDir, formatId, outPath] = process.argv.slice(2);
const { Dex, TeamValidator } = require(path.resolve(showdownDir, 'dist/sim'));

const format = Dex.formats.get(formatId, true);
if (!format.exists) throw new Error(`unknown format ${formatId}`);
const dex = Dex.forFormat(format);
const validator = TeamValidator.get(format);
const ruleTable = Dex.formats.getRuleTable(format);
const textEn = dex.loadTextData('en');
const textJa = dex.loadTextData('ja');
const text = (table, id) => {
  const en = textEn[table][id] || {}, ja = textJa[table][id] || {};
  return { shortDesc: en.shortDesc || '', desc: en.desc || '', nameJa: ja.name && ja.name !== en.name ? ja.name : null };
};
// Battle message templates (e.g. "{POKEMON}の **{MOVE}**！") used to parse OCR'd battle text.
const messages = { ja: {}, en: {} };
for (const lang of ['ja', 'en']) {
  const t = lang === 'ja' ? textJa : textEn;
  for (const table of ['Moves', 'Abilities', 'Items']) {
    for (const [id, entry] of Object.entries(t[table])) {
      const msgs = Object.fromEntries(Object.entries(entry).filter(([k, v]) => typeof v === 'string' && v.includes('{')));
      if (Object.keys(msgs).length) messages[lang][`${table}:${id}`] = msgs;
    }
  }
  messages[lang]['Default:default'] = t.Default.default;
  for (const [id, entry] of Object.entries(t.Default)) if (id !== 'default') messages[lang][`Default:${id}`] = entry;
}

const legalSpecies = dex.species.all().filter(s =>
  !s.isNonstandard && !validator.checkSpecies({}, s, s, {}) && !ruleTable.isBannedSpecies(s));

const species = legalSpecies.map(s => {
  const learnset = new Set();
  for (const move of dex.moves.all()) {
    if (move.isNonstandard && move.isNonstandard !== 'Future') continue;
    if (!validator.checkCanLearn(move, s, validator.allSources(s), {})) learnset.add(move.id);
  }
  return {
    id: s.id, name: s.name, nameJa: ((textJa.Pokedex[s.id] || {}).name !== s.name && (textJa.Pokedex[s.id] || {}).name) || null, num: s.num, baseSpecies: s.baseSpecies, forme: s.forme,
    types: s.types, baseStats: s.baseStats, abilities: Object.values(s.abilities),
    weightkg: s.weightkg, isMega: !!s.isMega, requiredItem: s.requiredItem || null,
    changesFrom: s.changesFrom || null, otherFormes: s.otherFormes || [],
    learnset: [...learnset].sort(),
  };
});

// Champions-new entries are tagged isNonstandard 'Future' in the base dex; keep whatever legal species use.
const usedMoves = new Set(species.flatMap(s => s.learnset));
const usedAbilities = new Set(species.flatMap(s => s.abilities.map(a => dex.abilities.get(a).id)));
const usedItems = new Set(legalSpecies.map(s => s.requiredItem).filter(Boolean).map(i => dex.items.get(i).id));
const moves = dex.moves.all().filter(m => !m.isNonstandard || usedMoves.has(m.id)).map(m => ({
  id: m.id, name: m.name, num: m.num, type: m.type, category: m.category,
  basePower: m.basePower, accuracy: m.accuracy, pp: m.pp, priority: m.priority,
  target: m.target, flags: Object.keys(m.flags), ...text('Moves', m.id),
  secondary: m.secondary || null, drain: m.drain || null, recoil: m.recoil || null,
  boosts: m.boosts || null, self: m.self || null, status: m.status || null,
  volatileStatus: m.volatileStatus || null, weather: m.weather || null,
  terrain: m.terrain || null, sideCondition: m.sideCondition || null,
  multihit: m.multihit || null, critRatio: m.critRatio || 1,
  selfSwitch: m.selfSwitch || null, forceSwitch: !!m.forceSwitch,
}));

const abilities = dex.abilities.all().filter(a => (!a.isNonstandard || usedAbilities.has(a.id)) && a.id !== 'noability').map(a => ({
  id: a.id, name: a.name, num: a.num, ...text('Abilities', a.id), rating: a.rating,
  flags: Object.keys(a.flags || {}),
  handlers: Object.keys(a).filter(k => /^on[A-Z]/.test(k)).sort(),
}));

const items = dex.items.all().filter(i => !i.isNonstandard || usedItems.has(i.id) ||
  (i.isNonstandard === 'Future' && !validator.validateSet({ species: 'Pikachu', item: i.name, moves: ['thunderbolt'], ability: 'static', evs: {}, level: 50 }, {})?.some(p => p.includes(i.name))))
  .map(i => ({
  id: i.id, name: i.name, num: i.num, ...text('Items', i.id),
  megaStone: i.megaStone || null, megaEvolves: i.megaEvolves || null,
  isBerry: !!i.isBerry, fling: i.fling || null,
  handlers: Object.keys(i).filter(k => /^on[A-Z]/.test(k)).sort(),
}));

const typechart = {};
for (const atk of dex.types.names()) {
  typechart[atk] = {};
  for (const def of dex.types.names()) typechart[atk][def] = dex.getEffectiveness(atk, def) === 0 && !dex.getImmunity(atk, def) ? 0 : 2 ** dex.getEffectiveness(atk, def);
}

const out = { format: format.id, mod: format.mod, exportedAt: new Date().toISOString(),
  rules: { pickedTeamSize: ruleTable.pickedTeamSize, maxTeamSize: ruleTable.maxTeamSize, level: ruleTable.adjustLevel },
  species, moves, abilities, items, typechart,
  typeNamesJa: textJa.TypeNames, natureNamesJa: textJa.NatureNames, messages };
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(`species=${species.length} moves=${moves.length} abilities=${abilities.length} items=${items.length}`);
