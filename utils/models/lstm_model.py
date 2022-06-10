import torch
from torch import nn

from .embedding import MetaModel

from ..training import activations


class LstmEncoderV1(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        dropout,
        num_layers,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm1(x.transpose(1, 2))
        return h_n[-1]


class LstmV1(nn.Module):
    def __init__(
        self,
        input_size,
        out_len,
        num_layers,
        channels,
        dropout,
        meta_cont_lens,
        meta_cat_lens,
        hd,
        meta_hd,
        activation,
        **_
    ):
        super().__init__()
        self.encoder = LstmEncoderV1(
            input_size,
            channels,
            dropout,
            num_layers,
        )

        self.meta = MetaModel(
            meta_cont_lens,
            meta_cat_lens,
            meta_hd,
            activation,
        )

        self.predict = nn.Sequential(
            nn.Linear(channels + meta_hd, hd),
            activations[activation](),
            nn.Linear(hd, out_len),
        )

    def forward(self, x, cont, cat):
        y = self.encoder(x)
        meta = self.meta(cont, cat)
        y = self.predict(torch.cat([y, meta], dim=1))

        return y


lstm_models = dict(
    LstmV1=LstmV1,
)
