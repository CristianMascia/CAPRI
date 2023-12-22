import json
import os

import numpy as np
import pandas as pd

import questions_utils as qu
import utils

######## Which is the best-performing CD algorithm in predicting anomalies?

dags_dir = "question0/dags/"
config_dir = "question0/configs/"
path_mets = "question0/metrics.json"
algorithms = ['dlingam', 'dagma_lin', 'dagma_mlp']

os.makedirs(dags_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)


def data_preparation():
    print("----------DATA PREPARATION----------")
    df = pd.read_csv("../mubench/data/mubench_df.csv")
    df_d, maps = utils.get_discovery_dataset(df)
    return df, df_d, maps


def discovery(df_discovery):
    print("----------CAUSAL DISCOVERY----------")

    maps = {i: df_discovery.columns[i] for i in range(0, len(df_discovery.columns))}
    X = df_discovery.to_numpy(dtype=np.float64)

    for cmod in algorithms:
        print("  -----" + cmod.upper() + "-----")
        qu.save_adjusted_dag(dags_dir + cmod, getattr(qu, cmod + "_discovery")(X, th=0), maps)


def configuration_generation(df_discovery, loads_mapping):
    print("----------CONFIGURATION GENERATION----------")

    for cmodel in algorithms:
        print("  -----" + cmodel.upper() + "-----")
        qu.gen_configs_per_metric(df_discovery, loads_mapping, dags_dir + cmodel + ".dot",
                                  config_dir + cmodel)


def show_metrics(df_all):
    print("----------METRICS----------")
    mets_models = {}
    for cmodel in algorithms:
        print("  -----" + cmodel.upper() + "-----")
        mets_models[cmodel] = {}
        for met in qu.all_metrics:
            mets_models[cmodel][met] = qu.calc_metrics(df_all, path_configs=config_dir + cmodel + "_" + met,
                                                       metrics=[met])
            print("{} -> PREC: {:.2f} RECALL: {:.2f} MEAN DIST: {:.2f} MIN DIST: {} MAX DIST: {}"
                  .format(met, mets_models[cmodel][met]['precision'], mets_models[cmodel][met]['recall'],
                          mets_models[cmodel][met]['mean_hamming_distance'],
                          mets_models[cmodel][met]['min_hamming_distance'],
                          mets_models[cmodel][met]['max_hamming_distance']))

    with open(path_mets, 'w') as f:
        json.dump(mets_models, f)


def __main__():
    df, df_dicovery, loads_map = data_preparation()
    discovery(df_dicovery)
    configuration_generation(df_dicovery, loads_map)
    show_metrics(df)

