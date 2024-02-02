import json
import os
import numpy as np
import pandas as pd
import questions_utils as qu
import utils
import CONFIG


######## Does prior knowledge improve performance?

def data_preparation(path_df):
    print("----------DATA PREPARATION----------")
    df = pd.read_csv(path_df)
    df_d, maps = utils.get_discovery_dataset(df)
    return df, df_d, maps


def discovery(df_discovery, dags_dir):
    print("----------CAUSAL DISCOVERY----------")

    maps = {i: df_discovery.columns[i] for i in range(0, len(df_discovery.columns))}
    X = df_discovery.to_numpy(dtype=np.float64)

    print("  -----DLINGAM WITH PRIOR-----")

    prior = qu.get_prior_mubench(df_discovery.columns)
    utils.draw_prior_knwoledge_mat(prior, df_discovery.columns, os.path.join(dags_dir, "prior"))
    qu.save_adjusted_dag(os.path.join(dags_dir, "dlingam_prior"), qu.dlingam_discovery(X, prior_knowledge=prior), maps)


def configuration_generation(df_discovery, loads_mapping, dags_dir, config_dir):
    print("----------CONFIGURATION GENERATION----------")

    print("  -----DLINGAM WITH PRIOR-----")
    qu.gen_configs_per_metric(df_discovery, loads_mapping, os.path.join(dags_dir, "dlingam_prior.dot"),
                              os.path.join(config_dir, "dlingam_prior"))


def show_metrics(df_all, config_dir, path_mets):
    print("----------METRICS----------")
    print("  -----DLINGAM_PRIOR-----")
    mets_model = qu.calc_metrics(df_all, config_dir, "dlingam_prior")

    with open(path_mets, 'w') as f:
        json.dump(mets_model, f)


def merge_mets(met_dicts):
    return qu.merge_met_dict(met_dicts)


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
