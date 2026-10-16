// Anytime information-set tree search. Legal root replies are covered before selective deepening.
// Simulation is omniscient; decisions in the tree and rollout only receive the acting player's observation.
const {createHash} = require('crypto');
const {PublicObservation} = require('./public_observation');
const {pruneRoot} = require('./action_filter');
const {choosePublic, chooseBatonPublic, observedStep, VERSION, BATON_VERSION} = require('./public_policy');
const hash = x => createHash('sha256').update(typeof x === 'string' ? x : JSON.stringify(x)).digest('hex');
let session = null;

class Branch extends Error {
  constructor(options) { super('enumerate chance'); this.options = options; }
}

// Exhaust all discrete chance calls, not just a handful of lucky seeds. A cap/timeout/unsupported draw
// yields UNKNOWN, never a certificate. This proves only the supplied sampled worlds, not all hidden sets.
function immediateWin(snapshot, a, c, deps, deadline, limit = 4096) {
  const queue = [[]]; let tested = 0;
  while (queue.length) {
    if (Date.now() >= deadline || ++tested > limit) return {win: false, complete: false, tested};
    const prefix = queue.pop(), b = deps.State.deserializeBattle(snapshot); let at = 0;
    const pick = options => {
      if (options.length === 1) return options[0];
      if (at === prefix.length) throw new Branch(options);
      const x = prefix[at++];
      if (!options.includes(x)) throw Error('chance path mismatch');
      return x;
    };
    b.prng.randomChance = (n, d) => pick(n <= 0 ? [false] : n >= d ? [true] : [false, true]);
    b.prng.random = (from, to) => {
      if (from === undefined) throw Error('continuous chance is not certified');
      const lo = to ? Math.floor(from) : 0, hi = Math.floor(to || from);
      if (hi - lo < 1 || hi - lo > 100) throw Error(`unsupported chance range ${lo}:${hi}`);
      return pick(Array.from({length: hi - lo}, (_, i) => lo + i));
    };
    try {
      deps.step(b, a, c);
      if (!b.ended || b.winner !== 'p1') return {win: false, complete: true, tested};
    } catch (e) {
      if (e instanceof Branch) for (const x of e.options) queue.push([...prefix, x]);
      else return {win: false, complete: false, tested, reason: String(e.message || e)};
    } finally { b.destroy(); }
  }
  return {win: true, complete: true, tested};
}

class Search {
  constructor(req, deps) {
    this.deps = deps; this.history = req.observationLog; this.nodes = new Map(); this.pending = new Map();
    this.policy = req.batonPolicy === false ? choosePublic : chooseBatonPublic;
    this.policyVersion = req.batonPolicy === false ? VERSION : BATON_VERSION;
    // stream: an independent random stream over the same worlds (offline oracles merge several; live uses none).
    this.id = hash(req.stream ? [req.samples, this.history, req.stream] : [req.samples, this.history]);
    this.serial = 0; this.cursor = 0; this.simulations = 0;
    this.failures = 0; this.errors = []; this.censored = 0; this.learned = 0; this.terminal = 0;
    this.maxTreeDepth = 0; this.maxRolloutTurns = 0; this.certified = []; this.chancePaths = 0;
    // Deep scripted wins against a weak rollout policy are not proofs of a winning position.
    // Spend the budget on repeated tactical comparisons, with a bounded public-value frontier.
    this.maxTurns = Math.min(200, Math.max(1, req.maxTurns ?? 8)); this.rootMix = null;
    // priorVisits: the V_ref bootstrap of a cell counts as this many rollouts, instead of being discarded after one.
    this.priorVisits = Math.max(0, Number(req.priorVisits ?? 4));
    this.priorWeight = this.priorVisits;
    this.minimumTrials = 2;
    // Rollout policies rotate per trial. The live default plays our Baton Pass plans against two scripted foes;
    // 'expert' for p2 keeps the rollouts symmetric (the foe sets up and finishes as well as we do).
    const rotation = (side, fallback) => {
      const list = req.rolloutPolicies && req.rolloutPolicies[side];
      if (!Array.isArray(list) || !list.length) return fallback;
      for (const name of list) if (!['greedy', 'tactical', 'expert', 'disrupt'].includes(name)) {
        throw Error(`unknown rollout policy ${name}`);
      }
      return list;
    };
    this.rolloutPolicies = {p1: rotation('p1', ['expert']), p2: rotation('p2', ['disrupt', 'tactical'])};
    const custom = req.rolloutPolicies && (req.rolloutPolicies.p1 || req.rolloutPolicies.p2);
    this.policyVersion = this.policyVersion + (custom ?
      `+p1:${this.rolloutPolicies.p1.join(',')}/p2:${this.rolloutPolicies.p2.join(',')}` : '');
    this.active = null; this.unfinishedAtDeadline = 0;
    this.states = req.samples.map(sample => {
      const b = deps.setUp(sample, req.format, 1);
      const st = {snapshot: JSON.stringify(deps.State.serializeBattle(b)), entry: b.pokesimEntryLog || [],
        p1: deps.choices(b, b.p1), p2: deps.choices(b, b.p2), lastFoe: b.p2.pokemonLeft === 1,
        speed: {ours: b.p1.active[0].getStat('spe'), opponent: b.p2.active[0].getStat('spe'),
          ourAction: b.p1.active[0].getActionSpeed(), opponentAction: b.p2.active[0].getActionSpeed()}};
      b.destroy(); return st;
    });
    if (!this.states.length || this.states.some(s => JSON.stringify(s.p1) !== JSON.stringify(this.states[0].p1))) {
      throw Error('terminal search needs common root choices');
    }
    this.actionFilter = pruneRoot(this.states, deps);
    for (const st of this.states) {
      st.cells = st.p1.map(() => st.p2.map(() => ({prior: null, n: 0, sum: 0, sumSq: 0, speed: 0, ends: 0, failed: 0})));
    }
    this.jobs = [];
    for (let i = 0; i < this.states[0].p1.length; i++) for (let w = 0; w < this.states.length; w++) {
      for (let j = 0; j < this.states[w].p2.length; j++) this.jobs.push([w, i, j]);
    }
  }

  certify(ms) {
    if (!this.states.every(s => s.lastFoe)) return;
    const deadline = Date.now() + ms;
    const cache = new Map();
    for (let i = 0; i < this.states[0].p1.length; i++) {
      const label = this.states[0].p1[i][1];
      if (!label.startsWith('move:')) continue;
      const probe = this.deps.State.deserializeBattle(this.states[0].snapshot);
      const attack = probe.dex.moves.get(label.split(':')[1]).category !== 'Status'; probe.destroy();
      if (!attack) continue;
      let win = true;
      for (const st of this.states) {
        for (const [c] of st.p2) {
          const key = hash([st.snapshot, st.p1[i][0], c]);
          let result = cache.get(key);
          if (!result) {
            result = immediateWin(st.snapshot, st.p1[i][0], c, this.deps, deadline);
            this.chancePaths += result.tested; cache.set(key, result);
          }
          if (!result.win) { win = false; break; }
        }
        if (!win) break;
      }
      if (win) { this.certified.push(label); return; }
      if (Date.now() >= deadline) return;
    }
  }

  observer(b, st) {
    const obs = new PublicObservation(this.history, b.dex, this.deps.extractChannelMessages,
      b.dex.currentMod.startsWith('champions'));
    obs.append(st.entry);
    for (const side of b.sides) {
      obs.bindActive(side.id, side.active[0]?.name); obs.bindActive(side.id, side.active[0]?.illusion?.name);
    }
    return obs;
  }

  // Each key is actor-visible JSON only. In particular p1's tree is shared across hidden-set samples.
  treeChoice(b, obs, side, trace) {
    const view = obs.view(side), options = this.deps.choices(b, side);
    const key = hash([view, options]), prior = this.policy(view, side.id === 'p1' ? 'expert' : 'disrupt', b.dex)[0];
    let node = this.nodes.get(key);
    if (!node) {
      node = {n: 0, actions: options.map(([choice]) => ({choice, n: 0, sum: 0}))};
      this.nodes.set(key, node);
    }
    let best = null, score = -Infinity;
    for (const action of node.actions) {
      const q = action.n ? action.sum / action.n + .8 * Math.sqrt(Math.log(node.n + 1) / action.n) :
        1e6 + (action.choice === prior ? 1 : 0);
      if (q > score) { score = q; best = action; }
    }
    trace.push({node, action: best, sign: side.id === 'p1' ? 1 : -1});
    return {choice: best.choice, fresh: best.n === 0};
  }

  backup(item, p) {
    const v = 2 * p - 1, cell = item.cell;
    if (item.initial) {
      // A root-step terminal result is exact in this sampled branch. Open positions retain their bounded
      // reference/material anchor instead of being replaced by the first deep rollout's binary result.
      cell.prior = v;
      cell.priorSpeed = item.ended && p === 1 ? 1 / (1 + item.turns) : 0;
      return;
    }
    cell.n++; cell.sum += v; cell.sumSq += v * v;
    if (item.ended) { cell.ends++; cell.speed += p === 1 ? 1 / (1 + item.turns) : 0; }
    for (const x of item.trace) { x.node.n++; x.action.n++; x.action.sum += x.sign * v; }
  }

  feedback(rows) {
    for (const row of rows || []) {
      const item = this.pending.get(row.id);
      if (!item || !Number.isFinite(row.p) || row.p < 0 || row.p > 1) throw Error('invalid tree feedback');
      this.backup(item, row.p); this.pending.delete(row.id); this.learned++;
    }
    if (this.pending.size) throw Error('missing tree feedback');
  }

  *trial(job, initial, output) {
    const [w, i, j] = job, st = this.states[w], cell = st.cells[i][j];
    const b = this.deps.State.deserializeBattle(st.snapshot), trace = [];
    const bytes = Buffer.from(hash([this.id, w, this.serial++]), 'hex');
    b.resetRNG([0, 2, 4, 6].map(x => bytes.readUInt16LE(x)));
    const obs = this.observer(b, st); let at = b.log.length, root = true, decisions = 0, tree = !initial;
    const start = b.turn;
    const names = this.rolloutPolicies, serial = this.serial;
    const rollout = side => this.policy(obs.view(side),
      names[side.id][serial % names[side.id].length], b.dex)[0];
    const advance = (a, c) => {
      this.deps.observedStep(b, a, c, this.deps.step, rollout, root ? [] : ['p1', 'p2']);
      root = false; obs.append(b.log.slice(at), true); at = b.log.length;
    };
    try {
      if (!b.ended) advance(st.p1[i][0], st.p2[j][0]);
      if (!initial && !b.ended && b.requestState === 'move') yield;
      while (!b.ended && (!initial || b.requestState !== 'move')) {
        if (++decisions > 600 || (b.requestState === 'move' &&
            b.turn - start >= this.maxTurns)) break;
        let a, c;
        if (tree) {
          // Choose independently from the same pre-action history, including separate forced switch phases.
          const x = this.treeChoice(b, obs, b.p1, trace), y = this.treeChoice(b, obs, b.p2, trace);
          a = x.choice; c = y.choice;
          this.maxTreeDepth = Math.max(this.maxTreeDepth, trace.length / 2);
          if (x.fresh || y.fresh) tree = false; // expand once, then use the frozen public rollout policy
        } else { a = rollout(b.p1); c = rollout(b.p2); }
        advance(a, c);
        if (!initial && !b.ended && b.requestState === 'move') yield;
      }
      const item = {cell, trace, initial, ended: b.ended, turns: b.turn - start};
      this.maxRolloutTurns = Math.max(this.maxRolloutTurns, item.turns);
      if (b.ended) {
        this.backup(item, b.winner === 'p1' ? 1 : b.winner === 'p2' ? 0 : .5); this.terminal++;
      } else {
        // V_ref is valid at normal move decisions, not halfway through a switch/queued turn.
        if (b.requestState !== 'move') throw Error('budget ended during forced switch; no value assigned');
        const id = this.serial; this.pending.set(id, item);
        output.push({id, observation: obs.view(b.p1)});
        if (!initial) this.censored++;
      }
      if (!initial) this.simulations++;
    } catch (e) {
      cell.failed++; this.failures++;
      if (this.errors.length < 5) this.errors.push(String(e.message || e));
    } finally { b.destroy(); }
  }

  pickJob() {
    // Every reply gets repeated evidence before a seemingly good row consumes the remaining budget.
    const underexplored = this.jobs.filter(([w, i, j]) => {
      const c = this.states[w].cells[i][j]; return c.n + c.failed < this.minimumTrials;
    });
    if (underexplored.length) return underexplored[this.simulations % underexplored.length];
    const n = this.states[0].p1.length, visits = Array(n).fill(0);
    for (const st of this.states) for (let i = 0; i < n; i++) visits[i] += st.cells[i].reduce((s, c) => s + c.n, 0);
    const total = visits.reduce((a, b) => a + b, 0), scores = visits.map((x, i) =>
      (this.rootMix?.[i] || 0) + Math.sqrt(2 * Math.log(total + 2) / (1 + x)));
    const row = scores.indexOf(Math.max(...scores));
    let job, least = Infinity;
    for (let w = 0; w < this.states.length; w++) for (let j = 0; j < this.states[w].p2.length; j++) {
      const c = this.states[w].cells[row][j], count = c.n + c.failed;
      if (count < least) { job = [w, row, j]; least = count; }
    }
    return job;
  }

  advance(ms, mix) {
    this.rootMix = mix || this.rootMix;
    const deadline = Date.now() + ms, pending = [];
    while (Date.now() < deadline && !this.certified.length) {
      if (!this.active) {
        this.activeOutput = [];
        this.active = this.cursor < this.jobs.length ?
          this.trial(this.jobs[this.cursor++], true, this.activeOutput) :
          this.trial(this.pickJob(), false, this.activeOutput);
      }
      if (this.active.next().done) {
        pending.push(...this.activeOutput); this.active = null;
      }
    }
    return this.reply(pending);
  }

  reply(pending = []) {
    const ready = this.states.every(st => st.cells.every(row => row.every(c => c.prior !== null))) ||
      this.certified.length > 0;
    const summary = {algorithm: 'public-information-set-uct-v2', particle_id: this.id,
      continuation_policy: this.policyVersion, bootstrap_reference: VERSION,
      root_speed_samples: this.states.map(s => s.speed),
      ready, simulations: this.simulations,
      terminal: this.terminal, learned: this.learned, censored: this.censored,
      failed: this.failures, errors: this.errors,
      tree_nodes: this.nodes.size, max_tree_decisions: this.maxTreeDepth, max_rollout_turns: this.maxRolloutTurns,
      horizon_turns: this.maxTurns, prior_weight: this.priorWeight, minimum_reply_trials: this.minimumTrials,
      value_kind: 'bounded-reference-material-utility-v1', calibrated_win_probability: false,
      action_filter: this.actionFilter,
      unfinished_at_deadline: this.unfinishedAtDeadline,
      chance_paths: this.chancePaths, immediate_wins: this.certified,
      certificate_scope: 'all discrete chance paths and legal replies in sampled worlds only'};
    return {session: this.id, pending, summary, results: this.states.map(st => ({
      p1: st.p1.map(x => x[1]), p2: st.p2.map(x => x[1]),
      payoff: st.cells.map((row, i) => row.map(c => this.certified.includes(st.p1[i][1]) ? 1 : this.value(c))),
      winSpeed: st.cells.map(row => row.map(c => this.speed(c))),
      visits: st.cells.map(row => row.map(c => c.n)), invalid: st.cells.map(row => row.map(c => c.failed)),
      stats: st.cells.map(row => row.map(c => ({n: c.n, sum: c.sum, sumSq: c.sumSq, prior: c.prior, speed: c.speed,
        priorSpeed: c.priorSpeed || 0, ends: c.ends, failed: c.failed}))),
      // A sampling-error indicator, not a confidence interval: rollouts share a policy and hidden-set particle.
      uncertainty: st.cells.map(row => row.map(c => {
        if (c.prior === null) return null;
        const n = this.priorWeight + c.n, mean = (this.priorWeight * c.prior + c.sum) / n;
        const second = (this.priorWeight * (c.prior * c.prior + .25) + c.sumSq) / n;
        return Math.sqrt(Math.max(0, second - mean * mean) / n);
      })),
      terminal_search: summary,
    }))};
  }

  // A cell's value: rollouts and the V_ref bootstrap weighted by priorVisits. With none, the first rollout
  // replaces the bootstrap; a single +-1 draw then stands for the cell and the planner's worst case runs on it.
  value(c) {
    if (c.prior === null) return null;
    return (this.priorVisits * c.prior + c.sum) / (this.priorVisits + c.n);
  }

  speed(c) {
    const n0 = c.prior === null ? 0 : this.priorVisits, p = c.priorSpeed || 0;
    if (!c.n) return p;
    return (n0 * p + c.speed) / (n0 + c.n);
  }

  finish() {
    // A final unfinished simulation is explicitly unscored. RPC time slices never create value labels.
    if (this.active) { this.active.return(); this.active = null; this.unfinishedAtDeadline++; }
    return this.reply();
  }
}

function terminalSearch(req, deps) {
  const op = req.terminalSearch;
  if (op.op === 'start') {
    session = new Search(req, {...deps, observedStep}); session.certify(Math.min(1200, op.certifyMs ?? 1200));
  } else if (!session || session.id !== op.session) throw Error('unknown terminal search session');
  session.feedback(op.feedback);
  if (op.op === 'finish') { const reply = session.finish(); session = null; return reply; }
  return session.advance(Math.min(1000, Math.max(1, op.sliceMs || 500)), op.mix);
}

module.exports = {terminalSearch, immediateWin, Search};
