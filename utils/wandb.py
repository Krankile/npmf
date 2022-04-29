import wandb as wb
import pandas as pd


def get_df_artifact(name: str, project: str):
    with wb.init(project=project) as run:
        art = run.use_artifact(name)
        art.download()
        filepath = art.file()

        return pd.read_feather(filepath)


def put_dataset(df: pd.DataFrame, filename: str, project: str, reset_index: bool = False):
    if reset_index:
        df = df.reset_index()

    df.to_feather(filename)

    artifact = wb.Artifact(filename.split(".")[0], type="dataset")
    artifact.add_file(filename)

    with wb.init(project=project) as run:
        run.log_artifact(artifact)