// Runs an exported PolicyNet (src/pokesim/learn/model.py export_json) on policy_features vectors, in plain JS.
// Loaded once per process and cached by path; `prior` gives the probability of each legal choice of a view and
// `sample` draws one with the trial's seeded random source, so rollouts stay reproducible.
const fs = require('fs');
const {createHash} = require('crypto');
const pf = require('./policy_features');
const schema = require('./policy_features.schema.json');
const cache = new Map();

function dense(layer) {
  const out = layer.w.length, inp = layer.w[0].length, w = new Float32Array(out * inp);
  for (let i = 0; i < out; i++) for (let j = 0; j < inp; j++) w[i * inp + j] = layer.w[i][j];
  return {w, b: Float32Array.from(layer.b), out, inp};
}

function apply(layer, x, relu) {
  const y = new Float32Array(layer.out);
  for (let i = 0; i < layer.out; i++) {
    let s = layer.b[i];
    const base = i * layer.inp;
    for (let j = 0; j < layer.inp; j++) s += layer.w[base + j] * x[j];
    y[i] = relu && s < 0 ? 0 : s;
  }
  return y;
}

class PolicyNet {
  constructor(path) {
    const text = fs.readFileSync(path, 'utf8');
    const data = JSON.parse(text);
    if (data.schema_version !== 1) throw Error('unsupported policy net schema');
    if (data.feature_version !== schema.version) throw Error('policy net was trained on other features');
    this.sha = createHash('sha256').update(text).digest('hex');
    this.head = data.head;
    this.decisionScope = data.decision_scope || 'all';
    if (!['all', 'move'].includes(this.decisionScope)) throw Error('unknown policy decision scope');
    this.trunk = data.trunk.map(dense); this.action = data.action.map(dense); this.scorer = data.scorer.map(dense);
    if (this.trunk[0].inp !== schema.state.length || this.action[0].inp !== schema.action.length) {
      throw Error('policy net feature sizes differ from the schema');
    }
  }

  logits(stateVec, actionVecs) {
    let s = stateVec;
    for (const layer of this.trunk) s = apply(layer, s, true);
    return actionVecs.map(a => {
      let h = apply(this.action[0], a, true);
      const joined = new Float32Array(s.length + h.length); joined.set(s); joined.set(h, s.length);
      let z = joined;
      for (let k = 0; k < this.scorer.length; k++) z = apply(this.scorer[k], z, k < this.scorer.length - 1);
      return z[0];
    });
  }

  // Probabilities of each legal (choice, label) pair of a view.
  supports(view) { return this.decisionScope === 'all' || view.phase === 'move'; }

  prior(view, dex, legal, temperature = 1) {
    if (!this.supports(view)) throw Error('policy does not support this decision phase');
    const state = pf.vector(pf.stateFeatures(view, dex), schema.state);
    const actions = legal.map(([choice, label]) => pf.vector(pf.actionFeatures(view, dex, choice, label), schema.action));
    const logits = this.logits(state, actions).map(x => x / temperature);
    const top = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - top)), total = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / total);
  }

  // One legal choice drawn by `random()` (a number in [0, 1)) from the prior.
  sample(view, dex, legal, random, temperature = 1) {
    const p = this.prior(view, dex, legal, temperature);
    let r = random(), acc = 0;
    for (let i = 0; i < legal.length; i++) { acc += p[i]; if (r < acc) return legal[i]; }
    return legal[legal.length - 1];
  }
}

function loadPolicyNet(path) {
  if (!cache.has(path)) cache.set(path, new PolicyNet(path));
  return cache.get(path);
}

module.exports = {PolicyNet, loadPolicyNet};
