import json
import os.path
import CONFIG
import data_preparation
import sys
import importlib


def __main__():
    n_reps = 10
    n_questions = 4

    path_system = os.path.join("..", "mubench")
    path_exps = os.path.join(path_system, "data")
    path_df = os.path.join(path_exps, "mubench_df.csv")

    if not os.path.exists(path_df):
        data_preparation.read_experiments(path_exps, {s: s for s in CONFIG.services}, CONFIG.services,
                                          data_preparation.rename_startwith).to_csv(path_df, index=False)
    for i in range(1, n_questions + 1):
        print("QUESTION {}".format(i))
        q = "question{}".format(i)
        mets_dicts = [{} for j in range(n_reps)]

        for rep in range(n_reps):
            print("REPETITION {}".format(rep))
            path_dir = os.path.join(q, "{}_REP_{}".format(q, rep))
            path_mets = os.path.join(path_dir, "metrics.json")
            importlib.import_module(q)
            getattr(sys.modules[q], "__main__")(path_df, path_dir)
            with open(path_mets, 'r') as f_met:
                mets_dicts[rep] = json.load(f_met)

        with open(os.path.join(q, "mean_metrics_{}.json".format(q)), 'w') as f_mean:
            json.dump(getattr(sys.modules[q], "merge_mets")(mets_dicts), f_mean)


__main__()
