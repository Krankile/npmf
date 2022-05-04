import wandb as wb
import pandas as pd


def get_dataset(name: str, project: str):
    with wb.init(project=project) as run:
        art = run.use_artifact(name)
        art.download()
        filepath = art.file()

        return pd.read_feather(filepath)


def put_dataset(
    df: pd.DataFrame,
    filename: str,
    project: str,
    type_: str = "dataset",
    drop_index: bool = True,
    description: str = None,
):
    df.reset_index(drop=drop_index).to_feather(filename)

    artifact = wb.Artifact(filename.split(".")[0], type=type_, description=description)
    artifact.add_file(filename)

    with wb.init(project=project) as run:
        run.log_artifact(artifact)
