"""Two-level Deep Sets: encode moves before pooling within each Pokemon, then encode the roster.

Pooling raw Pokemon attributes loses which recipient has a status or which move has PP.
Shared nonlinear encoders keep those associations, independently of slot ordering.
"""
import torch
from torch import nn


class RosterEncoder(nn.Module):
    def __init__(self, layout, hidden):
        super().__init__()
        self.layout = layout
        self.species = nn.Embedding(layout['species_count'], 12, padding_idx=0)
        self.moves = nn.Embedding(layout['move_count'], 12, padding_idx=0)
        self.move_encoder = nn.Sequential(nn.Linear(layout['move_width'] - 1 + 12, 32), nn.Tanh(),
                                          nn.Linear(32, 32), nn.Tanh())
        self.mon_encoder = nn.Sequential(nn.Linear(layout['move_start'] - 1 + 12 + 32, hidden), nn.Tanh(),
                                         nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, tokens):
        split, width = self.layout['move_start'], self.layout['move_width']
        moves = tokens[..., split:].reshape(*tokens.shape[:-1], 4, width)
        move_ids = moves[..., 1].long()
        numeric = torch.cat((moves[..., :1], moves[..., 2:]), dim=-1)
        encoded = self.move_encoder(torch.cat((numeric, self.moves(move_ids)), dim=-1))
        encoded = (encoded * moves[..., :1]).sum(-2) / 4
        numeric = torch.cat((tokens[..., :2], tokens[..., 3:split]), dim=-1)
        mons = self.mon_encoder(torch.cat((numeric, self.species(tokens[..., 2].long()), encoded), dim=-1))
        return (mons * tokens[..., :1]).sum(-2) / 3


class SetValueModel(nn.Module):
    def __init__(self, public_dim, private_dim, hidden=128, sets=None):
        super().__init__()
        self.config = dict(public_dim=public_dim, private_dim=private_dim, hidden=hidden,
                           sets=sets, architecture='nested-sets-v2')
        self.layout = sets
        embed = hidden // 2
        self.public_roster = RosterEncoder(sets, embed)
        self.private_roster = RosterEncoder(sets, embed)
        preview = next(b['width'] for b in sets['blocks'] if 'preview' in b['label'])
        self.preview_species = nn.Embedding(sets['species_count'], 12, padding_idx=0)
        self.preview_encoder = nn.Sequential(nn.Linear(preview - 1 + 12, 32), nn.Tanh())
        self.public_encoder = nn.Sequential(nn.Linear(sets['public_context'] + 4 + 2 * embed + 64, hidden),
                                            nn.Tanh(), nn.Linear(hidden, embed), nn.Tanh())
        self.public_head = nn.Linear(embed, 1)
        self.private_head = nn.Sequential(nn.Linear(embed + sets['private_context'] + embed, embed),
                                          nn.Tanh(), nn.Linear(embed, 1))
        nn.init.zeros_(self.private_head[-1].weight)
        nn.init.zeros_(self.private_head[-1].bias)

    def blocks(self, data, branch):
        result = []
        for b in self.layout['blocks']:
            if b['branch'] != branch:
                continue
            tokens = data[:, b['start']:b['start'] + b['count'] * b['width']].reshape(
                len(data), b['count'], b['width'])
            if 'preview' in b['label']:
                numeric = torch.cat((tokens[..., :1], tokens[..., 2:]), dim=-1)
                value = self.preview_encoder(torch.cat((numeric, self.preview_species(tokens[..., 1].long())), -1))
                result.append((value * tokens[..., :1]).sum(1) / 6)
            else:
                result.append((self.public_roster if branch == 'public' else self.private_roster)(tokens))
        return result

    def forward(self, public, private=None):
        context = public[:, :self.layout['public_context']]
        encoded = self.public_encoder(torch.cat((context, *self.blocks(public, 'public'), public[:, -4:]), -1))
        value = self.public_head(encoded).squeeze(-1)
        if private is not None:
            inputs = torch.cat((encoded, private[:, :self.layout['private_context']],
                                *self.blocks(private, 'private')), -1)
            # A missing private request is not a fictitious all-zero Pokemon roster.
            value = value + self.private_head(inputs).squeeze(-1) * private[:, 0]
        return value
