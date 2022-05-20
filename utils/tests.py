import pickle


def unpickle_df(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_df(df, filename):
    with open(filename, "wb") as f:
        pickle.dump(df, f)
