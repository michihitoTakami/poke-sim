// Offline diagnostics for the state boundary used by the 2026-10-18 audit.
// Run from a checkout with its engine/data installed, or pass that checkout as argv[2].
// Hypothetical sets and fixed actions isolate state effects; these are not live win-rate estimates.
const path = require('path');
const fs = require('fs');
const root = path.resolve(process.argv[2] || process.cwd());
const dir = process.env.POKESIM_SHOWDOWN_DIR || path.join(root, 'external/pokemon-showdown');
const {setUp} = require(path.join(root, 'services/sim-tools/state_codec.js')).createCodec(dir);
const own = JSON.parse(fs.readFileSync(path.join(root, 'data/my_teams/baton.json'), 'utf8'));
const format = 'gen9championsbssregmc';
const king = {
  species: 'Kingambit', ability: 'Supreme Overlord', item: 'Black Glasses', nature: 'Adamant',
  moves: ['Kowtow Cleave', 'Sucker Punch', 'Iron Head', 'Swords Dance'],
  evs: {hp: 32, atk: 32, spe: 2}, level: 50,
};
const clefable = {
  species: 'Clefable', ability: 'Magic Guard', item: '', level: 50,
  moves: ['Moonblast', 'Flamethrower', 'Stealth Rock', 'Moonlight'],
};
const pawmot = {
  species: 'Pawmot', ability: 'Volt Absorb', item: '', level: 50,
  moves: ['Nuzzle', 'Revival Blessing', 'Close Combat', 'Thunder Punch'],
};
const hippo = {
  species: 'Hippowdon', ability: 'Sand Stream', item: 'Leftovers', level: 50,
  moves: ['Earthquake', 'Yawn', 'Slack Off', 'Stealth Rock'],
};
const sample = {
  p1: {team: [own[4], own[3], own[1]], mons: [
    {hp: 140 / 211, boosts: {spa: 2}}, {hp: 1}, {hp: 26 / 175, status: 'par'},
  ]},
  p2: {team: [king, clefable, pawmot], mons: [{hp: 1}, {hp: 0}, {hp: 0}]},
  field: {},
};
const supremeOverlord = [];
for (const fallen of [null, 3]) {
  const b = setUp(sample, format, 19), mon = b.p2.active[0];
  // The explicit control supplies the published state that the current codec omits.
  if (fallen !== null) {
    b.p2.totalFainted = fallen;
    mon.abilityState.fallen = fallen;
  }
  const before = {
    totalFainted: b.p2.totalFainted, abilityFallen: mon.abilityState.fallen ?? null,
    pokemonLeft: b.p2.pokemonLeft,
  };
  b.choose('p1', 'move slackoff');
  b.choose('p2', 'move kowtowcleave');
  supremeOverlord.push({explicitFallen: fallen, before, ourHPAfter: b.p1.pokemon[0].hp});
  b.destroy();
}
const yawn = [];
for (const remaining of [null, 1]) {
  const us = {hp: 184 / 201, volatiles: ['yawn'], lastMove: 'snarl'};
  if (remaining !== null) us.volatileTurns = {yawn: remaining};
  const b = setUp({
    p1: {team: [own[5], own[4], own[3]], mons: [us, {hp: 1}, {hp: 0}]},
    p2: {team: [hippo], mons: [{hp: .98}]}, field: {weather: 'sandstorm'},
  }, format, 19);
  const before = b.p1.active[0].volatiles.yawn?.duration;
  b.choose('p1', 'move moonlight');
  b.choose('p2', 'move earthquake');
  yawn.push({
    explicitRemaining: remaining, durationBefore: before,
    durationAfter: b.p1.active[0].volatiles.yawn?.duration ?? null,
    statusAfter: b.p1.active[0].status,
  });
  b.destroy();
}
const substitute = [];
for (const alreadyDrowsy of [false, true]) {
  const us = {hp: 92 / 173, mega: true};
  if (alreadyDrowsy) {
    us.volatiles = ['yawn'];
    us.volatileTurns = {yawn: 1};
  }
  const b = setUp({
    p1: {team: [own[3]], mons: [us]},
    p2: {team: [hippo], mons: [{hp: .68}]}, field: {weather: 'sandstorm'},
  }, format, 19);
  b.choose('p1', 'move substitute');
  b.choose('p2', alreadyDrowsy ? 'move slackoff' : 'move yawn');
  substitute.push({
    alreadyDrowsy, hpAfter: b.p1.active[0].hp,
    substituteHP: b.p1.active[0].volatiles.substitute?.hp ?? null,
    yawnAfter: !!b.p1.active[0].volatiles.yawn,
    statusAfter: b.p1.active[0].status,
    events: b.log.filter(l => l.startsWith('|-fail') || l.startsWith('|-status')),
  });
  b.destroy();
}
console.log(JSON.stringify({
  scope: 'offline state diagnostics with hypothetical sets; no live orders',
  format, seed: 19, samples: {supremeOverlord: sample, hippo},
  supremeOverlord, yawn, substitute,
}, null, 1));
