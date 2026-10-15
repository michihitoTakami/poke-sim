"""Validated batched V_ref inference for observation-constrained search leaves."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from pokesim.evaluation.branch_matches import file_sha
from pokesim.value.features import Encoder
from pokesim.value.model import load_checkpoint


class ValueEnsemble:
    def __init__(self, run):
        run = Path(run).resolve()
        report = json.loads((run / 'report.json').read_text())
        cfg = report['provenance']
        data = Path(cfg['settings']['data'])
        meta = json.loads((data / 'manifest.json').read_text())
        if (file_sha(data / 'manifest.json') != cfg['data_manifest_sha'] or
                meta['dataset_id'] != cfg['dataset_id'] or
                meta['feature_code_sha'] != file_sha(Path(__file__).with_name('features.py')) or
                file_sha(data / 'dex.json') != cfg['dex_sha']):
            raise ValueError('value features or dex differ from training')
        expected = {'own': 'expert', 'opponent_pool': ['expert/tactical', 'expert/disrupt'],
                    'version': 'public-reference-v2'}
        if cfg['reference'] != expected:
            raise ValueError('unsupported value continuation contract')
        self.encoder = Encoder.load(data / 'dex.json')
        self.features = cfg['features']
        self.models, hashes = [], {}
        torch.set_num_threads(2)
        for seed in cfg['settings']['seeds']:
            item = report['stages'][f'pretrained:{seed}']
            path = run / item['checkpoint']
            if path.parent != run or file_sha(path) != item['sha']:
                raise ValueError('modified value checkpoint')
            model, saved = load_checkpoint(path, cfg['dataset_id'])
            if saved['reference'] != expected or saved['stage'] != 'pretrained' or saved['features'] != self.features:
                raise ValueError('value checkpoint contract mismatch')
            self.models.append(model)
            hashes[path.name] = item['sha']
        if not self.models:
            raise ValueError('empty value ensemble')
        self.version = {'dataset_id': cfg['dataset_id'], 'report_sha': file_sha(run / 'report.json'),
                        'checkpoints': hashes, 'reference': expected, 'run': str(run)}

    def probabilities(self, observations):
        if not observations:
            return np.asarray([], np.float32)
        public, private = [], []
        for obs in observations:
            p, q, _, _, schema = self.encoder.encode(obs)
            if schema != self.features:
                raise ValueError('value observation schema mismatch')
            if obs['side'] != 'p1' or not q[0]:
                raise ValueError('value requires own request from the normalized p1 perspective')
            public.append(p)
            private.append(q)
        p, q = torch.from_numpy(np.stack(public)), torch.from_numpy(np.stack(private))
        with torch.no_grad():
            result = torch.stack([torch.sigmoid(model(p, q)) for model in self.models]).mean(0).numpy()
        if not np.isfinite(result).all():
            raise ValueError('nonfinite value prediction')
        return result

    def apply(self, reply):
        start = time.perf_counter()
        if reply.get('policyVersion') != self.version['reference']['version']:
            raise ValueError('worker/value policy mismatch')
        leaves = [leaf for res in reply['results'] for leaf in res['leaves'] if leaf['terminal'] is None]
        predictions = iter(self.probabilities([leaf['observation'] for leaf in leaves]))
        for res in reply['results']:
            shape = (len(res['p1']), len(res['p2']))
            sums, counts, resolved, hp = [np.zeros(shape) for _ in range(4)]
            per = {}
            n_terminal = 0
            for leaf in res.pop('leaves'):
                i, j, name = leaf['i'], leaf['j'], leaf['continuation']
                terminal = leaf['terminal']
                value = 2 * (float(next(predictions)) if terminal is None else terminal) - 1
                sums[i, j] += value
                hp[i, j] += leaf['hpDiagnostic']
                counts[i, j] += 1
                resolved[i, j] += terminal is not None
                n_terminal += terminal is not None
                if name not in per:
                    per[name] = (np.zeros(shape), np.zeros(shape))
                per[name][0][i, j] += value
                per[name][1][i, j] += 1

            def mean(total, number):
                return [[float(total[i, j] / number[i, j]) if number[i, j] else None
                         for j in range(shape[1])] for i in range(shape[0])]

            res.update(payoff=mean(sums, counts), resolved=mean(resolved, counts),
                       byContinuation={name: mean(a, n) for name, (a, n) in per.items()},
                       hpDiagnostic=mean(hp, counts),
                       value_status={'terminal': int(n_terminal), 'learned': int(counts.sum() - n_terminal),
                                     'failed': int(np.asarray(res['invalid']).sum()),
                                     'model': self.version, 'inference_ms': 1000 * (time.perf_counter() - start)})
        return reply['results']
