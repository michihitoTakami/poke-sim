// Battle-ready details of a team (actual stats, max HP, move PP, details string) computed by the simulator.
// Usage: node stats.js <showdown dir> <format> < team.txt   (team in Showdown export format)
// Output: [{"name", "species", "details", "maxhp", "stats": {atk, def, spa, spd, spe}, "ability", "item", "megaStone",
//           "moves": [{"id", "move", "pp", "maxpp", "target"}]}, ...]
const path = require('path');
const [showdownDir, format] = process.argv.slice(2);
const { Battle, Teams } = require(path.resolve(showdownDir, 'dist/sim'));

let input = '';
process.stdin.on('data', d => { input += d; });
process.stdin.on('end', () => {
  const team = Teams.pack(Teams.import(input));
  const battle = new Battle({ formatid: format, seed: [1, 2, 3, 4] });
  battle.setPlayer('p1', { name: 'me', team });
  battle.setPlayer('p2', { name: 'them', team });
  const out = battle.p1.pokemon.map(p => ({
    name: p.name,
    species: p.species.name,
    details: p.details,
    maxhp: p.maxhp,
    stats: p.storedStats,
    ability: p.ability,
    item: p.item,
    megaStone: !!battle.dex.items.get(p.item).megaStone,
    moves: p.moveSlots.map(m => ({ id: m.id, move: m.move, pp: m.pp, maxpp: m.maxpp, target: m.target })),
  }));
  process.stdout.write(JSON.stringify(out));
});
