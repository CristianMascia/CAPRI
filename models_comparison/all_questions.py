import os.path
import CONFIG
import data_preparation
import sys
import importlib


def __main__():
    path_df = "../mubench/data/mubench_df.csv"
    if not os.path.exists(path_df):
        data_preparation.read_experiments("../mubench/data/", {s: s for s in CONFIG.services}).to_csv(path_df,
                                                                                                      index=False)
    for i in range(4):
        q = "question{}".format(i)
        importlib.import_module(q)
        getattr(sys.modules[q], "__main__")(path_df, q)


__main__()
