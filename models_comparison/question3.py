import json
import os
import lingam
import numpy as np
import pandas as pd
import questions_utils as qu
import utils

######## How does performance vary with changing the minimum confidence required for a causal relation identification?

dags_dir = "question3/dags/"
config_dir = "question3/configs/"
path_mets = "question3/metrics.json"

ths = [0.1, 0.3]
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

    model = lingam.DirectLiNGAM(
        prior_knowledge=utils.get_generic_priorknorledge_mat(df_discovery.columns, qu.services, qu.path_wm,
                                                             num_load=len(qu.loads)))
    model.fit(X)
    adj_mat = np.transpose(model.adjacency_matrix_)

    for th in ths:
        print("  -----DLINGAM WITH PRIOR TH: {}-----".format(th))
        qu.save_dag(dags_dir + "dlingam_prior_th" + str(th), utils.threshold_matrix(adj_mat, th=th), maps)


def configuration_generation(df_discovery, loads_mapping):
    print("----------CONFIGURATION GENERATION----------")

    for th in ths:
        print("  -----TH:" + str(th) + "-----")

        qu.gen_configs_per_metric(df_discovery, loads_mapping, dags_dir + "dlingam_prior_th" + str(th) + ".dot",
                                  config_dir + "TH_" + str(th))


def show_metrics(df_all):
    print("----------METRICS----------")
    mets_models = {}
    for th in ths:
        print("  -----TH:" + str(th) + "-----")
        mets_models[str(th)] = {}
        for met in qu.all_metrics:
            mets_models[str(th)][met] = qu.calc_metrics(df_all, path_configs=config_dir + "TH_" + str(th),
                                                        metrics=[met])
            print("{} -> PREC: {:.2f} RECALL: {:.2f} MEAN DIST: {:.2f} MIN DIST: {} MAX DIST: {}"
                  .format(met, mets_models[str(th)][met]['precision'], mets_models[str(th)][met]['recall'],
                          mets_models[str(th)][met]['mean_hamming_distance'],
                          mets_models[str(th)][met]['min_hamming_distance'],
                          mets_models[str(th)][met]['max_hamming_distance']))

    with open(path_mets, 'w') as f:
        json.dump(mets_models, f)


def __main__():
    df, df_dicovery, loads_map = data_preparation()
    discovery(df_dicovery)
    configuration_generation(df_dicovery, loads_map)
    show_metrics(df)
