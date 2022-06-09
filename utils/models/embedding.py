from operator import itemgetter

import torch
from torch import nn

from ..training import activations


class MetaModel(nn.Module):
    def __init__(
        self,
        meta_cont_lens,
        meta_cat_lens,
        meta_hd,
        activation,
    ):
        super().__init__()

        self.meta_cont = nn.Sequential(
            nn.Linear(meta_cont_lens[0], meta_cont_lens[1]),
            activations[activation](),
        )

        self.meta_cat = nn.ModuleList(
            [nn.Embedding(in_, out) for in_, out in meta_cat_lens]
        )

        self.meta_hidden = nn.Sequential(
            nn.Linear(
                meta_cont_lens[1] + sum(map(itemgetter(1), meta_cat_lens)),
                meta_hd,
            ),
            activations[activation](),
        )


    def forward(self, cont, cat):
        return self.meta_hidden(
            torch.cat(
                [self.meta_cont(cont)]
                + [emb(cat[:, i]) for i, emb in enumerate(self.meta_cat)],
                dim=1,
            )
        )