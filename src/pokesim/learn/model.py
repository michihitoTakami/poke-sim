"""A small policy net over state and per-action features, exportable to plain JSON for the JS worker."""
import json
from pathlib import Path

import torch
from torch import nn


class PolicyNet(nn.Module):
    """score(state, action) = w · relu(W2 relu(W1 [trunk(state); act(action)])) ; softmax over legal actions.
    Everything is Linear + ReLU so services/sim-tools/policy_net.js can run it with a few matrix products."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 192, embed: int = 128, action_hidden: int = 64,
                 scorer_hidden: int = 64):
        super().__init__()
        self.config = {'state_dim': state_dim, 'action_dim': action_dim, 'hidden': hidden, 'embed': embed,
                       'action_hidden': action_hidden, 'scorer_hidden': scorer_hidden}
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_dim, action_hidden), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(embed + action_hidden, scorer_hidden), nn.ReLU(),
                                    nn.Linear(scorer_hidden, 1))

    def forward(self, state, actions, mask):
        """state [B, S], actions [B, A, D], mask [B, A] (bool) -> masked logits [B, A]."""
        s = self.trunk(state)  # [B, E]
        a = self.action(actions)  # [B, A, H]
        joined = torch.cat([s.unsqueeze(1).expand(-1, a.shape[1], -1), a], dim=-1)
        logits = self.scorer(joined).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)

    def export(self) -> dict:
        """Weights as nested lists: layers in evaluation order, each {w: [out][in], b: [out]}."""
        def layer(linear: nn.Linear) -> dict:
            return {'w': linear.weight.detach().cpu().tolist(), 'b': linear.bias.detach().cpu().tolist()}
        return {'config': self.config,
                'trunk': [layer(self.trunk[0]), layer(self.trunk[2])],
                'action': [layer(self.action[0])],
                'scorer': [layer(self.scorer[0]), layer(self.scorer[2])]}


def save_checkpoint(path: Path, model: PolicyNet, extra: dict) -> None:
    torch.save({'schema_version': 1, 'model_config': model.config, 'state_dict': model.state_dict(), **extra}, path)


def load_checkpoint(path: Path) -> tuple[PolicyNet, dict]:
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema_version') != 1:
        raise ValueError('policy checkpoint schema mismatch')
    model = PolicyNet(**saved['model_config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved


def export_json(path: Path, model: PolicyNet, meta: dict) -> None:
    Path(path).write_text(json.dumps({'schema_version': 1, **meta, **model.export()}))
