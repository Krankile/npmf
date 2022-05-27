import pickle
from typing import Iterable

import wandb as wb
import pandas as pd


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
            art = run.use_artifact(name); art.download()
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


def put_models(filename: str, model_dict: dict, metadata: dict = None):
    with open(filename, mode="wb") as f:
        pickle.dump(model_dict, f)

    with wb.init(project="master-test") as run:
        art = wb.Artifact(filename.split(".")[0], type="model", metadata=metadata)
        art.add_file(filename)

        run.log_artifact(art)


def get_models(artifact_name: str):
    with wb.init(project="master-test") as run:
        art = run.use_artifact(artifact_name)
        art.download()
        filename = art.file()

    with open(filename, mode="rb") as f:
        return pickle.load(f)


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
