from pathlib import Path
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


def get_datasets(names: Iterable[str], project: str, run=None):
    r = run if run is not None else wb.init(project=project)

    dfs = []
    for name in names:
        art = r.use_artifact(name)
        art.download()
        df = pd.read_feather(art.file())
        dfs.append(df)

    if run is None:
        r.finish()

    return dfs


def put_dataset(
    df: pd.DataFrame,
    filename: str,
    project: str,
    type_: str = "dataset",
    drop_index: bool = True,
    description: str = None,
    metadata: dict = None,
    run=None,
):
    df.reset_index(drop=drop_index).to_feather(filename)

    artifact = wb.Artifact(
        filename.split(".")[0], type=type_, description=description, metadata=metadata
    )
    artifact.add_file(filename)

    r = run if run is not None else wb.init(project=project)
    r.log_artifact(artifact)

    if run is None:
        r.finish()


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


def get_nn_model(artifact_name: str, project: str, run=None) -> Tuple[nn.Module, dict]:
    r = run if run is not None else wb.init(project=project)

    artifact = r.use_artifact(f"krankile/{project}/{artifact_name}", type="model")
    artifact.download()
    model_state_dict = artifact.file()
    conf: dict = artifact.metadata
    model: nn.Module = models[conf["model"]](**conf)
    model.load_state_dict(
        torch.load(
            model_state_dict,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
    if run is None:
        r.finish()

    return model, conf


def put_nn_model(model: nn.Module, run) -> None:
    filename = f"model-{run.name}.pth"
    conf: wb.Config = run.config

    torch.save(model.state_dict(), filename)
    art = wb.Artifact(conf.model, type="model", metadata=conf.as_dict())
    art.add_file(filename)

    run.log_artifact(art)


data_artifacts = {
    20: "era-datasets:v3",
    240: "era-datasets:v4",
}


def get_processed_data(run, kind):
    artifact = data_artifacts[kind]
    path = Path("./artifacts") / artifact

    art = run.use_artifact(artifact)

    if path.exists():
        return path

    art.download()

    return path
