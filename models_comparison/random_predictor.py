import json
import os
import random

import numpy as np
import pandas as pd
import questions_utils as qu
import utils
import CONFIG


######## Random Predictor

def data_preparation(path_df):
    print("----------DATA PREPARATION----------")
    return pd.read_csv(path_df)


def configuration_generation(df, config_dir):
    print("----------CONFIGURATION GENERATION----------")

    nusers = list(set(df['NUSER']))
    loads = list(set(df['LOAD']))
    SRs = list(set(df['SR']))

    for ser in CONFIG.services:
        for met in CONFIG.all_metrics:
            conf = {"nusers": [random.choice(nusers)], "loads": [random.choice(loads)],
                    "spawn_rates": [random.choice(SRs)], "anomalous_metrics": [met]}
            with open(os.path.join(config_dir, "random_predictor_{}_{}.json".format(met, ser)), 'w') as f_conf:
                json.dump(conf, f_conf)


def show_metrics(df_all, config_dir, path_mets):
    print("----------METRICS----------")

    print("  -----RANDOM PREDICTOR-----")
    mets = qu.calc_metrics(df_all, config_dir, "random_predictor")

    with open(path_mets, 'w') as f:
        json.dump(mets, f)
    return mets


def __main__(path_df, path_main_dir):
    mets = [{}] * 5
    for rep in range(5):
        path_dir_configs = os.path.join(path_main_dir, "random_predictor_REP_{}".format(rep), "configs")
        path_metric_file = os.path.join(path_main_dir, "random_predictor_REP_{}".format(rep), "metrics.json")

        os.makedirs(path_dir_configs, exist_ok=True)

        df = data_preparation(path_df)

        # configuration_generation(df, path_dir_configs)
        mets[rep] = show_metrics(df, path_dir_configs, path_metric_file)

    with open(os.path.join(path_main_dir, "mean_metrics_random_predictor.json"), 'w') as f_mean_mets:
        json.dump(qu.merge_met_dict(mets), f_mean_mets)


__main__(CONFIG.path_df, "random_predictor")
