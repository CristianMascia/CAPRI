import os.path
import CONFIG
import data_preparation
import sys
import importlib


def __main__():
    n_reps = 3
    n_questions = 4

    path_df = "../mubench/data/mubench_df.csv"
    if not os.path.exists(path_df):
        data_preparation.read_experiments("../mubench/data/", {s: s for s in CONFIG.services}).to_csv(path_df,
                                                                                                      index=False)

    for rep in range(n_reps):
        print("REPETITION {}".format(rep))
        for i in range(n_questions):
            q = "question{}".format(i)
            importlib.import_module(q)
            getattr(sys.modules[q], "__main__")(path_df, "{}_REP_{}".format(q, rep))



