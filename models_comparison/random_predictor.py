import json
import os
import random

import numpy as np
import pandas as pd

import example
import questions_utils as qu
import utils
import CONFIG


######## Random Predictor

def __main__(path_df, path_main_dir):
    df = pd.read_csv(path_df)

    mets = [{}] * 5
    for rep in range(5):
        path_dir = os.path.join(path_main_dir, "random_predictor_REP_{}".format(rep))
        path_dir_configs = os.path.join(path_main_dir, "random_predictor_REP_{}".format(rep), "generated_configs")
        path_metric_file = os.path.join(path_main_dir, "random_predictor_REP_{}".format(rep), "metrics.json")

        os.makedirs(path_dir, exist_ok=True)

        example.random_predictor(example.System.MUBENCH, path_dir)
        mets[rep] = qu.calc_metrics(df, path_dir_configs, "configs")

        with open(path_metric_file, 'w') as f:
            json.dump(mets[rep], f)

    with open(os.path.join(path_main_dir, "mean_metrics_random_predictor.json"), 'w') as f_mean_mets:
        json.dump(qu.merge_met_dict(mets), f_mean_mets)


__main__(CONFIG.path_df, "random_predictor")
