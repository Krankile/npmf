import pickle
from typing import Iterable, Tuple

import pandas as pd
import torch
from torch import nn

import wandb as wb

from .models import models


def get_dataset(name: str, project: str):
    with wb.init(project=project) as run:
        art = run.use_artifact(name)
        art.download()
        filepath = art.file()

        return pd.read_feather(filepath)


def get_datasets(names: Iterable[str], project: str):
    dfs = []
    with wb.init(project=project) as run:
        for name in names:
            art = run.use_artifact(name)
            art.download()
            df = pd.read_feather(art.file())
            dfs.append(df)
    return dfs


def put_dataset(
    df: pd.DataFrame,
    filename: str,
    project: str,
    type_: str = "dataset",
    drop_index: bool = True,
    description: str = None,
    metadata: dict = None,
):
    df.reset_index(drop=drop_index).to_feather(filename)

    artifact = wb.Artifact(
        filename.split(".")[0], type=type_, description=description, metadata=metadata
    )
    artifact.add_file(filename)

    with wb.init(project=project) as run:
        run.log_artifact(artifact)


def put_stat_models(
    filename: str, model_dict: dict, metadata: dict = None, project="master-test"
):
    with open(filename, mode="wb") as f:
        pickle.dump(model_dict, f)

    with wb.init(project=project) as run:
        art = wb.Artifact(filename.split(".")[0], type="model", metadata=metadata)
        art.add_file(filename)

        run.log_artifact(art)


def get_stat_models(artifact_name: str, project="master-test", metadata=False):
    with wb.init(project=project) as run:
        art = run.use_artifact(artifact_name)
        art.download()
        filename = art.file()

    with open(filename, mode="rb") as f:
        models = pickle.load(f)

    if metadata:
        return models, art.metadata

    return models


def update_aliases(project: str, alias: str, artifacts: Iterable):
    api = wb.Api()
    for artifact_ in artifacts:
        artifact = api.artifact(f"krankile/{project}/{artifact_}")
        if alias in artifact.aliases:
            artifact.aliases.remove(alias)
            print(f"alias: {alias} removed from {artifact_}")
        else:
            artifact.aliases.append(alias)
    artifact.save()


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


class TcnV2(nn.Module):
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
            nn.ReLU(),
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
            nn.Linear(channels + meta_hd, hd),
            nn.ReLU(),
            nn.Linear(hd, hd),
            nn.ReLU(),
            nn.Linear(hd, hd),
            nn.ReLU(),
            nn.Linear(hd, out_len),
        )
        # self.init_weights()

    def init_weights(self):
        self.predict.weight.data.normal_(0, 0.01)

    def forward(self, x, cont, cat):
        y = self.tcn(x)

        meta = self.meta_hidden(
            torch.cat(
                [self.meta_cont(cont)]
                + [emb(cat[:, i]) for i, emb in enumerate(self.meta_cat)],
                dim=1,
            )
        )

        y = self.predict(torch.cat([y[:, :, -1], meta], dim=1))

        return y


tcn_models = dict(
    TcnV1=TcnV1,
)
def get_tcnn_model(artifact_name: str, project: str) -> Tuple[nn.Module, dict]:
    with wb.init(project=project) as run:
        artifact = run.use_artifact(f"krankile/{project}/{artifact_name}", type="model")
        artifact.download()
        model_state_dict = artifact.file()
        conf: dict = artifact.metadata
        model: nn.Module = TcnV2(**conf)
        model.load_state_dict(
            torch.load(
                model_state_dict,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
    return model, conf

def get_nn_model(artifact_name: str, project: str) -> Tuple[nn.Module, dict]:
    with wb.init(project=project) as run:
        artifact = run.use_artifact(f"krankile/{project}/{artifact_name}", type="model")
        artifact.download()
        model_state_dict = artifact.file()
        conf: dict = artifact.metadata
        model: nn.Module = models[conf["model"]](**conf)
        model.load_state_dict(
            torch.load(
                model_state_dict,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
    return model, conf


def put_nn_model(model: nn.Module, run) -> None:
    filename = f"model-{run.name}.pth"
    conf: wb.Config = run.config

    torch.save(model.state_dict(), filename)
    art = wb.Artifact(conf.model, type="model", metadata=conf.as_dict())
    art.add_file(filename)

    run.log_artifact(art)
