import json
import os

import numpy as np
import pandas as pd

import questions_utils as qu
import utils

######## Does prior knowledge improve performance?

dags_dir = "question1/dags/"
config_dir = "question1/configs/"
path_mets = "question1/metrics.json"

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

    print("  -----DLINGAM WITH PRIOR-----")
    prior = utils.get_generic_priorknorledge_mat(df_discovery.columns, qu.services, qu.path_wm, num_load=len(qu.loads))
    qu.save_adjusted_dag(dags_dir + "dlingam_prior", qu.dlingam_discovery(X, prior_knowledge=prior), maps)


def configuration_generation(df_discovery, loads_mapping):
    print("----------CONFIGURATION GENERATION----------")

    print("  -----DLINGAM WITH PRIOR-----")
    qu.gen_configs_per_metric(df_discovery, loads_mapping, dags_dir + "dlingam_prior.dot", config_dir + "dlingam_prior")


def show_metrics(df_all):
    print("----------METRICS----------")
    print("  -----DLINGAM_PRIOR-----")
    mets_model = {}
    for met in qu.all_metrics:
        mets_model[met] = qu.calc_metrics(df_all, path_configs=config_dir + "dlingam_prior_" + met, metrics=[met])
        print("{} -> PREC: {:.2f} RECALL: {:.2f}".format(met, mets_model[met]['precision'], mets_model[met]['recall']))

    with open(path_mets, 'w') as f:
        json.dump(mets_model, f)


def __main__():
    df, df_dicovery, loads_map = data_preparation()
    discovery(df_dicovery)
    configuration_generation(df_dicovery, loads_map)
    show_metrics(df)


__main__()
