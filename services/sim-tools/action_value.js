// Offline comparison actor: conditional terminal Q values with a fixed ensemble disagreement guard.
const fs = require('fs'), path = require('path'), {createHash} = require('crypto');
const {loadPolicyNet} = require('./policy_net');
const pf = require('./policy_features'), schema = require('./policy_features.schema.json');
const {legalChoices} = require('./public_observation');
const cache = new Map();

function guardedIndex(probabilities, baseline, margin = .05, disagreement = 1.64) {
  const n = probabilities.length;
  const lower = probabilities[0].map((_, i) => {
    const d = probabilities.map(p => p[i] - p[baseline]);
    const mean = d.reduce((a, b) => a + b, 0) / n;
    const variance = d.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    return mean - disagreement * Math.sqrt(variance);
  });
  const best = lower.indexOf(Math.max(...lower));
  return lower[best] > margin ? best : baseline;
}

function loadActionValue(run) {
  if (cache.has(run)) return cache.get(run);
  const text = fs.readFileSync(path.join(run, 'report.json'), 'utf8'), report = JSON.parse(text);
  if (report.version !== 'conditional-terminal-action-value-v1') throw Error('wrong action-value target');
  const nets = Object.entries(report.models).map(([name, sha]) => {
    if (path.basename(name) !== name || !name.endsWith('.json')) throw Error('invalid action-value path');
    const file = path.join(run, name), net = loadPolicyNet(file);
    if (net.sha !== sha || net.head !== 'action_value' || net.decisionScope !== 'move')
      throw Error('action-value model mismatch');
    const transform = JSON.parse(fs.readFileSync(file, 'utf8')).score_transform || 'sigmoid';
    if (!['identity', 'sigmoid'].includes(transform)) throw Error('invalid action-value score transform');
    return {net, transform};
  });
  if (nets.length < 2) throw Error('action-value comparison needs an ensemble');
  const model = {sha: createHash('sha256').update(text).digest('hex'), choose(view, dex, reference) {
    if (view.phase !== 'move') return reference;
    const legal = legalChoices(view), baseline = legal.findIndex(([c]) => c === reference[0]);
    if (baseline < 0) throw Error('reference choice missing from actor observation');
    const state = pf.vector(pf.stateFeatures(view, dex), schema.state);
    const actions = legal.map(([c, l]) => pf.vector(pf.actionFeatures(view, dex, c, l), schema.action));
    const probabilities = nets.map(({net, transform}) => net.logits(state, actions).map(
      x => transform === 'identity' ? x : 1 / (1 + Math.exp(-x))));
    return legal[guardedIndex(probabilities, baseline, report.guard.margin, report.guard.disagreement)];
  }};
  cache.set(run, model);
  return model;
}

module.exports = {guardedIndex, loadActionValue};
