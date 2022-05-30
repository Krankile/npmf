from operator import itemgetter

import torch
from torch import nn
from torch.nn.utils import weight_norm


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

        self.macro = nn.Sequential(
            nn.Linear(macro_len_in, macro_len_out),
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
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, cat):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


models = dict(
    NaiveMultivariateNetworkV1=NaiveMultivariateNetworkV1,
    TcnV1=TcnV1,
)
