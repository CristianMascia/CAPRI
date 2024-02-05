import json
import os
import lingam
import numpy as np
import pandas as pd
import questions_utils as qu
import utils
import CONFIG

######## How does performance vary with changing the minimum confidence required for a causal relation identification?


ths = [0.1, 0.3]


def data_preparation(path_df):
    print("----------DATA PREPARATION----------")
    df = pd.read_csv(path_df)
    df_d, maps = utils.get_discovery_dataset(df)
    return df, df_d, maps


def discovery(df_discovery, dags_dir):
    print("----------CAUSAL DISCOVERY----------")

    maps = {i: df_discovery.columns[i] for i in range(0, len(df_discovery.columns))}
    X = df_discovery.to_numpy(dtype=np.float64)

    prior = qu.get_prior_mubench(df_discovery.columns)
    utils.draw_prior_knwoledge_mat(prior, df_discovery.columns, os.path.join(dags_dir, "prior"))
    model = lingam.DirectLiNGAM(prior_knowledge=prior)
    model.fit(X)
    adj_mat = np.transpose(model.adjacency_matrix_)

    for th in ths:
        print("  -----DLINGAM WITH PRIOR TH: {}-----".format(th))
        qu.save_dag(os.path.join(dags_dir, "dlingam_prior_th" + str(th)), utils.threshold_matrix(adj_mat, th=th), maps)


def configuration_generation(df_discovery, loads_mapping, dags_dir, config_dir):
    print("----------CONFIGURATION GENERATION----------")

    for th in ths:
        print("  -----TH:" + str(th) + "-----")
        qu.gen_configs_per_metric(df_discovery, loads_mapping,
                                  os.path.join(dags_dir, "dlingam_prior_th{}.dot".format(th)),
                                  os.path.join(config_dir, "TH_" + str(th)))


def show_metrics(df_all, config_dir, path_mets):
    print("----------METRICS----------")
    mets_models = {}
    for th in ths:
        print("  -----TH:" + str(th) + "-----")
        mets_models[str(th)] = qu.calc_metrics(df_all, config_dir, "TH_" + str(th))

    with open(path_mets, 'w') as f:
        json.dump(mets_models, f)


def merge_mets(met_dicts):
    out_dict = {}
    for th in ths:
        out_dict[str(th)] = qu.merge_met_dict([m[str(th)] for m in met_dicts])

    return out_dict


def __main__(path_df, path_main_dir):
    path_dir_dags = os.path.join(path_main_dir, "dags")
    path_dir_configs = os.path.join(path_main_dir, "configs")
    path_metric_file = os.path.join(path_main_dir, "metrics.json")

    os.makedirs(path_dir_dags, exist_ok=True)
    os.makedirs(path_dir_configs, exist_ok=True)

    df, df_discovery, loads_map = data_preparation(path_df)

    discovery(df_discovery, path_dir_dags)
    configuration_generation(df_discovery, loads_map, path_dir_dags, path_dir_configs)
    show_metrics(df, path_dir_configs, path_metric_file)
