import torch
from torch import nn
from torch import Tensor as T
from operator import itemgetter


class NaiveTcnV1(nn.Module):
    def __init__(
        self,
        input_size,
        out_len,
        channels,
        kernel_size,
        meta_cont_lens,
        meta_cat_lens,
        hd,
        meta_hd,
        **_
    ):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=channels,
                kernel_size=kernel_size,
            ),
            nn.ReLU(),
        )

        self.meta_cont = nn.Sequential(
            nn.Linear(meta_cont_lens[0], meta_cont_lens[1]),
        )

        self.meta_cat = nn.ModuleList(
            [nn.Embedding(in_, out) for in_, out in meta_cat_lens]
        )

        self.meta_hidden = nn.Sequential(
            nn.Linear(
                meta_cont_lens[1] + sum(map(itemgetter(1), meta_cat_lens)),
                meta_hd,
            ),
            nn.ReLU(),
        )

        self.predict = nn.Sequential(
            nn.LazyLinear(channels + meta_hd, hd),
            nn.ReLU(),
            nn.Linear(hd, out_len),
        )

    def meta_embedding(self, cont, cat):
        return self.meta_hidden(
            torch.cat(
                [self.meta_cont(cont)]
                + [emb(cat[:, i]) for i, emb in enumerate(self.meta_cat)],
                dim=1,
            )
        )

    def forward(self, x, cont, cat):
        y: T = self.tcn(x)
        y = y.flatten(start_dim=1)

        meta = self.meta_embedding(cont, cat)
        y = self.predict(torch.cat([y, meta], dim=1))

        return y


naive_tcn_models = dict(
    NaiveTcnV1=NaiveTcnV1,
)