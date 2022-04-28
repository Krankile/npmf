import wandb as wb
import pandas as pd


def get_df_artifact(name):
    with wb.init(project="master-test") as run:
        art = run.use_artifact(name)
        art.download()
        filepath = art.file()

        return pd.read_feather(filepath)
