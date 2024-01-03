import json
import os
import numpy as np
import pandas as pd
import questions_utils as qu
import utils
import CONFIG

######## Which is the best-performing CD algorithm in predicting anomalies?

algorithms = ['dlingam', 'dagma_lin', 'dagma_mlp', 'dag_gnn']


def data_preparation(path_df):
    print("----------DATA PREPARATION----------")
    df = pd.read_csv(path_df)
    df_d, maps = utils.get_discovery_dataset(df)
    return df, df_d, maps


def discovery(df_discovery, dags_dir):
    print("----------CAUSAL DISCOVERY----------")

    maps = {i: df_discovery.columns[i] for i in range(0, len(df_discovery.columns))}
    X = df_discovery.to_numpy(dtype=np.float64)

    for cmod in algorithms:
        print("  -----" + cmod.upper() + "-----")
        path_mod = os.path.join(dags_dir, cmod)
        if cmod == 'dag_gnn':
            qu.dag_gnn_discovery(df_discovery, path_mod)
        else:
            qu.save_adjusted_dag(path_mod, getattr(qu, cmod + "_discovery")(X, th=0), maps)


def configuration_generation(df_discovery, loads_mapping, dags_dir, config_dir):
    print("----------CONFIGURATION GENERATION----------")

    for cmodel in algorithms:
        print("  -----" + cmodel.upper() + "-----")
        qu.gen_configs_per_metric(df_discovery, loads_mapping, os.path.join(dags_dir, cmodel + ".dot"),
                                  os.path.join(config_dir, cmodel))


def show_metrics(df_all, config_dir, path_mets):
    print("----------METRICS----------")
    mets_models = {}
    for cmodel in algorithms:
        print("  -----" + cmodel.upper() + "-----")
        mets_models[cmodel] = {}
        for met in CONFIG.all_metrics:
            mets_models[cmodel][met] = qu.calc_metrics(df_all,
                                                       path_configs=os.path.join(config_dir,
                                                                                 "{}_{}".format(cmodel, met)),
                                                       metrics=[met])
            print("{} -> PREC: {:.2f} RECALL: {:.2f} MEAN DIST: {:.2f} MIN DIST: {} MAX DIST: {}"
                  .format(met, mets_models[cmodel][met]['precision'], mets_models[cmodel][met]['recall'],
                          mets_models[cmodel][met]['mean_hamming_distance'],
                          mets_models[cmodel][met]['min_hamming_distance'],
                          mets_models[cmodel][met]['max_hamming_distance']))

    with open(path_mets, 'w') as f:
        json.dump(mets_models, f)


def merge_mets(met_dicts):
    out_dict = {}
    for alg in algorithms:
        out_dict[alg] = qu.merge_met_dict([m[alg] for m in met_dicts])

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
