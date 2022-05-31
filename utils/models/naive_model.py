from operator import itemgetter

import torch
from torch import nn


class NaiveMultivariateNetworkV1(nn.Module):
    def __init__(
        self,
        sf_len_in,
        sf_len_out,
        meta_cont_lens,
        meta_cat_lens,
        meta_hd,
        macro_len_in,
        macro_len_out,
        hd1,
        hd2,
        out_len,
        d1,
        d2,
        d3,
        **_,
    ):
        super().__init__()

        self.sf = nn.Sequential(
            nn.Linear(sf_len_in, sf_len_out),
            nn.ReLU(),
        )

        self.macro = nn.Sequential(
            nn.Linear(macro_len_in, macro_len_out),
            nn.ReLU(),
        )

        self.meta_cont = nn.Sequential(
            nn.Linear(meta_cont_lens[0], meta_cont_lens[1]),
            nn.ReLU(),
        )

        self.meta_cat = nn.ModuleList(
            [nn.Embedding(in_, out) for in_, out in meta_cat_lens]
        )

        self.meta_hidden = nn.Sequential(
            nn.Linear(
                meta_cont_lens[1] + sum(map(itemgetter(1), meta_cat_lens)), meta_hd
            ),
            nn.ReLU(),
        )

        total_width = sum([sf_len_out, macro_len_out, meta_hd])

        self.predict = nn.Sequential(
            nn.Dropout(p=d1),
            nn.Linear(total_width, hd1),
            nn.ReLU(),
            nn.Dropout(p=d2),
            nn.Linear(hd1, hd2),
            nn.ReLU(),
            nn.Dropout(p=d3),
            nn.Linear(hd2, out_len),
        )

    def forward(self, sf, meta_cont, meta_cat, macro):

        sf = self.sf(sf)
        meta = torch.cat(
            [self.meta_cont(meta_cont)]
            + [emb(meta_cat[:, i]) for i, emb in enumerate(self.meta_cat)],
            dim=1,
        )
        meta = self.meta_hidden(meta)
        macro = self.macro(macro)

        x = torch.cat((sf, meta, macro), dim=1)
        x = self.predict(x)

        return x


naive_models = dict(
    NaiveMultivariateNetworkV1=NaiveMultivariateNetworkV1,
)
