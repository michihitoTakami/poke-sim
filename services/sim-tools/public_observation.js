// The actor receives client-visible JSON only. No Battle/Pokemon objects or sampled opponent sets cross this API.
const clone = x => JSON.parse(JSON.stringify(x));
const id = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const effectId = s => id(String(s || '').replace(/^(?:move|ability|item): /, ''));

function health(text) {
  const [hp, status = ''] = text.split(' '), [n, d = 100] = hp.split('/').map(parseFloat);
  return {hp: n || 0, maxhp: d, fraction: (n || 0) / d, status};
}

function publicHealth(text, champions) {
  const h = health(text);
  if (!h.hp || h.maxhp === 100) return text;
  let pct = champions ? Math.max(1, Math.floor(h.fraction * 100)) : Math.ceil(h.fraction * 100);
  if (!champions && pct === 100 && h.hp < h.maxhp) pct = 99;
  let colour = '';
  if (champions && pct === 20) colour = h.hp * 5 > h.maxhp ? 'y' : 'r';
  if (champions && pct === 50) colour = h.hp * 2 > h.maxhp ? 'g' : 'y';
  return `${pct}/100${colour}${h.status ? ' ' + h.status : ''}`;
}

// Public preview/history is provided by the observed record, never inferred from the sampled selected roster.
class PublicObservation {
  constructor(log, dex, extract, champions = true) {
    if (!Array.isArray(log) || !log.some(line => /^\|turn\|\d+$/.test(line))) {
      throw new Error('public continuations need observed history through the root turn');
    }
    this.dex = dex;
    this.extract = extract;
    this.champions = champions;
    this.state = {turn: 0, preview: {p1: [], p2: []}, active: {p1: null, p2: null},
                  mons: {p1: {}, p2: {}}, weather: '', terrain: '', conditions: {p1: {}, p2: {}}};
    this.history = [];
    this.names = {p1: {}, p2: {}};
    this.pendingBaton = {p1: false, p2: false};
    this.entries = {p1: null, p2: null};
    this.offset = 0;
    this.append(log);
    this.offset = this.state.turn - 1;  // reconstructed simulator starts at turn 1
  }

  key(species) {
    let s = this.dex.species.get(species);
    if (typeof s.battleOnly === 'string') s = this.dex.species.get(s.battleOnly);
    return id(s.baseSpecies || s.name);
  }

  bindActive(side, name) {
    // Reconstruction may replace a player's nickname. Alias the active slot only, without revealing species/sets.
    const key = this.state.active[side];
    if (key && name) this.names[side][name] = key;
  }

  append(lines, simulated = false) {
    const publicLines = this.extract(lines.join('\n'), [0])[0];
    for (let line of publicLines) {
      let p = line.split('|'), kind = p[1];
      if (!kind || ['request', 'showteam', 'input', 'seed', 'debug', 'error', 't:', 'player', 'callback',
                    'sentchoice', 'choice', 'inactive', 'inactiveoff'].includes(kind)) continue;
      // The source log was captured from p1 and contains its exact HP. Both actors' shared history uses public HP.
      if (['switch', 'drag', 'replace'].includes(kind) && p[4]) p[4] = publicHealth(p[4], this.champions);
      if (['-damage', '-heal', '-sethp'].includes(kind) && p[3]) p[3] = publicHealth(p[3], this.champions);
      if (kind === '-sethp' && p[5]) p[5] = publicHealth(p[5], this.champions);
      if (kind === 'turn') p[2] = String(Number(p[2]) + (simulated ? this.offset : 0));
      line = p.join('|');
      this.history.push(line);
      if (kind === 'turn') {
        this.state.turn = Number(p[2]);
        this.pendingBaton = {p1: false, p2: false};
        continue;
      }
      if (kind === '-clearallboost') {
        for (const side of ['p1', 'p2']) for (const mon of Object.values(this.state.mons[side])) mon.boosts = {};
      }
      if (kind === 'clearpoke') { this.state.preview = {p1: [], p2: []}; continue; }
      if (kind === 'poke' && this.state.preview[p[2]]) { this.state.preview[p[2]].push(p[3]); continue; }
      if (kind === '-weather') { this.state.weather = p[2] === 'none' ? '' : p[2]; continue; }
      if (kind === '-fieldstart' && /terrain/i.test(p[2])) this.state.terrain = p[2];
      if (kind === '-fieldend' && /terrain/i.test(p[2])) this.state.terrain = '';
      const side = (p[2] || '').slice(0, 2);
      if (!this.state.mons[side]) continue;
      if (kind === '-sidestart') this.state.conditions[side][p[3]] = (this.state.conditions[side][p[3]] || 0) + 1;
      if (kind === '-sideend') delete this.state.conditions[side][p[3]];
      const name = (p[2] || '').split(': ')[1];
      if (kind === 'replace') {
        // Illusion's reveal is an identity correction, not a switch; the protocol may omit HP.
        const entry = this.entries[side], disguised = this.state.mons[side][this.state.active[side]];
        if (!entry || !disguised) throw new Error('replace without a public active entry');
        const key = this.key(p[3].split(',')[0]);
        if (entry.known) this.state.mons[side][entry.key] = clone(entry.known);
        else delete this.state.mons[side][entry.key];
        const known = this.state.mons[side][key] || {};
        this.state.mons[side][key] = {...known, ...entry.revealed, species: p[3].split(',')[0],
          condition: p[4] || disguised.condition, boosts: {...disguised.boosts},
          volatiles: disguised.volatiles.slice(), lastMove: disguised.lastMove,
          moves: [...new Set([...(known.moves || []), ...entry.moves])]};
        this.names[side][name] = key;
        this.state.active[side] = key;
        this.entries[side] = {key, known: clone(known), moves: entry.moves, revealed: entry.revealed};
        continue;
      }
      if (['switch', 'drag'].includes(kind)) {
        const key = this.key(p[3].split(',')[0]);
        const previous = this.state.mons[side][this.state.active[side]];
        const passing = this.pendingBaton[side] && previous;
        const passedBoosts = passing ? {...previous.boosts} : {};
        const passedVolatiles = passing ? previous.volatiles.filter(v => !this.dex.conditions.get(v).noCopy) : [];
        this.pendingBaton[side] = false;
        if (previous) { previous.boosts = {}; previous.volatiles = []; }
        this.names[side][name] = key;
        const old = this.state.mons[side][key] || {};
        this.entries[side] = {key, known: this.state.mons[side][key] ? clone(old) : null, moves: [], revealed: {}};
        this.state.mons[side][key] = {...old, species: p[3].split(',')[0], condition: p[4], boosts: passedBoosts,
                                    volatiles: passedVolatiles, moves: old.moves || [], lastMove: null};
        this.state.active[side] = key;
        continue;
      }
      const mon = this.state.mons[side][this.names[side][name]];
      if (!mon) continue;
      if (['detailschange', '-formechange'].includes(kind)) mon.species = p[3].split(',')[0];
      if (kind === 'faint') mon.condition = '0 fnt';
      if (['-damage', '-heal', '-sethp'].includes(kind)) mon.condition = p[3];
      if (kind === '-status') mon.condition = mon.condition.split(' ')[0] + ' ' + p[3];
      if (kind === '-curestatus') mon.condition = mon.condition.split(' ')[0];
      if (kind === 'move' && !line.includes('[from]')) {
        mon.lastMove = id(p[3]);
        this.pendingBaton[side] = mon.lastMove === 'batonpass';
        if (!mon.moves.includes(mon.lastMove)) mon.moves.push(mon.lastMove);
        if (this.entries[side] && !this.entries[side].moves.includes(mon.lastMove)) this.entries[side].moves.push(mon.lastMove);
      }
      if (kind === '-fail' || kind === 'faint') this.pendingBaton[side] = false;
      if (kind === '-item') mon.item = effectId(p[3]);
      if (kind === '-enditem') mon.item = '';
      if (kind === '-ability') mon.ability = effectId(p[3]);
      if (this.entries[side] && ['-item', '-enditem', '-ability'].includes(kind)) {
        const field = kind === '-ability' ? 'ability' : 'item';
        this.entries[side].revealed[field] = mon[field];
      }
      if (kind === '-boost') mon.boosts[p[3]] = Math.min(6, (mon.boosts[p[3]] || 0) + Number(p[4]));
      if (kind === '-unboost') mon.boosts[p[3]] = Math.max(-6, (mon.boosts[p[3]] || 0) - Number(p[4]));
      if (kind === '-setboost') mon.boosts[p[3]] = Number(p[4]);
      if (kind === '-clearboost') mon.boosts = {};
      if (kind === '-start' && !mon.volatiles.includes(effectId(p[3]))) mon.volatiles.push(effectId(p[3]));
      if (kind === '-end') {
        mon.volatiles = mon.volatiles.filter(x => x !== effectId(p[3]));
        if (effectId(p[3]) === 'substitute') mon.substituteBrokenTurn = this.state.turn;
      }
    }
  }

  view(side) {
    const request = clone(side.activeRequest || {wait: true});
    request.side = clone(side.getRequestData());  // the acting player's private information is legitimate
    return {schema: 1, side: side.id, turn: this.state.turn,
            phase: request.wait ? 'wait' : request.forceSwitch ? 'switch' : 'move',
            request, public: clone(this.state), history: this.history.slice()};
  }
}

function legalChoices(view) {
  const req = view.request, mons = req.side.pokemon;
  if (req.wait) return [['pass', 'pass']];
  const out = [];
  if (!req.forceSwitch) {
    (req.active?.[0]?.moves || []).forEach((m, i) => {
      if (m.disabled) return;
      out.push([`move ${i + 1}`, `move:${m.id}`]);
      if (req.active[0].canMegaEvo) out.push([`move ${i + 1} mega`, `move:${m.id}:mega`]);
    });
  }
  if (req.forceSwitch || !req.active?.[0]?.trapped) mons.forEach((m, i) => {
    if (!m.active && health(m.condition).hp > 0) out.push([`switch ${i + 1}`, `switch:${id(m.details.split(',')[0])}`]);
  });
  return out;
}

module.exports = {PublicObservation, legalChoices, health, id};
