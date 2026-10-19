"""Small public encoder and a separate private-information residual for fixed-reference value prediction."""
import torch
from torch import nn


class ValueModel(nn.Module):
    def __init__(self, public_dim, private_dim, hidden=96):
        super().__init__()
        self.config = {'public_dim': public_dim, 'private_dim': private_dim, 'hidden': hidden}
        self.encoder = nn.Sequential(nn.Linear(public_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden // 2), nn.Tanh())
        self.public_head = nn.Linear(hidden // 2, 1)
        self.private_head = nn.Sequential(nn.Linear(hidden // 2 + private_dim, hidden // 2), nn.Tanh(),
                                          nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.private_head[-1].weight)
        nn.init.zeros_(self.private_head[-1].bias)

    def forward(self, public, private=None):
        encoded = self.encoder(public)
        result = self.public_head(encoded).squeeze(-1)
        if private is not None:
            result = result + self.private_head(torch.cat((encoded, private), dim=-1)).squeeze(-1)
        return result


def predict(model, data, device='cpu', private=True, batch=4096):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data['target']), batch):
            p = torch.as_tensor(data['public'][start:start + batch], device=device)
            q = torch.as_tensor(data['private'][start:start + batch], device=device) if private else None
            predictions.append(torch.sigmoid(model(p, q)).cpu())
    return torch.cat(predictions).numpy()


def load_checkpoint(path, dataset_id=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema_version') != 1 or (dataset_id and saved['dataset_id'] != dataset_id):
        raise ValueError('checkpoint schema or feature dataset mismatch')
    model = model_from_config(saved['model_config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved


def model_from_config(config):
    config = dict(config)
    architecture = config.pop('architecture', 'mlp-v1')
    if architecture == 'nested-sets-v2':
        from pokesim.value.set_model import SetValueModel

        return SetValueModel(**config)
    if architecture != 'mlp-v1':
        raise ValueError('unsupported value model architecture')
    return ValueModel(**config)


def create_model(features, hidden):
    config = dict(public_dim=len(features['public']), private_dim=len(features['private']), hidden=hidden)
    if 'sets' in features:
        config.update(architecture='nested-sets-v2', sets=features['sets'])
    return model_from_config(config)
