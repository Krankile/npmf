from operator import itemgetter

import torch
from torch import nn
from torch.nn.utils import weight_norm

from ..training import activations


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TcnV1(nn.Module):
    def __init__(
        self,
        input_size,
        out_len,
        num_layers,
        channels,
        kernel_size,
        dropout,
        meta_cont_lens,
        meta_cat_lens,
        hd,
        meta_hd,
        activation,
        **_
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            input_size,
            [channels] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
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

        self.predict = nn.Sequential(
            nn.Linear(channels + meta_hd, hd),
            activations[activation](),
            nn.Linear(hd, out_len),
        )

        # self.init_weights()

    def init_weights(self):
        self.predict.weight.data.normal_(0, 0.01)

    def meta_embedding(self, cont, cat):
        return self.meta_hidden(
            torch.cat(
                [self.meta_cont(cont)]
                + [emb(cat[:, i]) for i, emb in enumerate(self.meta_cat)],
                dim=1,
            )
        )

    def forward(self, x, cont, cat):
        y = self.tcn(x)

        meta = self.meta_embedding(cont, cat)

        y = self.predict(torch.cat([y[:, :, -1], meta], dim=1))

        return y


class TcnV2(TcnV1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, cont, cat):
        y = super().forward(x, cont, cat)

        return y + 1


class TcnV3(TcnV1):
    def __init__(self, channels, meta_hd, hd, activation, out_len, **kwargs):
        super().__init__(
            channels=channels,
            meta_hd=meta_hd,
            hd=hd,
            activation=activation,
            out_len=out_len,
            **kwargs
        )

        self.tcn_steps = 10

        self.predict = nn.Sequential(
            nn.Linear(channels * self.tcn_steps + meta_hd, hd),
            activations[activation](),
            nn.Linear(hd, out_len),
        )

    def forward(self, x, cont, cat):
        meta = self.meta_embedding(cont, cat)
        y = self.tcn(x)[:, :, : self.tcn_steps].flatten(startdim=1)
        y = self.predict(torch.cat([y, meta], dim=1))

        return y


tcn_models = dict(
    TcnV1=TcnV1,
    TcnV2=TcnV2,
    TcnV3=TcnV3,
)
