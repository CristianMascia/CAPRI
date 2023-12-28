import json
import os
import numpy as np
import pandas as pd
import questions_utils as qu
import utils
import CONFIG

######## How does performance vary if we consider one model per performance metric or one model for all metrics?


def data_preparation(path_df):
    print("----------DATA PREPARATION----------")
    df = pd.read_csv(path_df)
    df_d, maps = utils.get_discovery_dataset(df)
    cols_base_all = ['NUSER', 'LOAD', 'SR'] + ['REQ/s_' + s for s in CONFIG.services]
    cols_base_disc = ['NUSER'] + ['LOAD_' + str(i) for i in range(3)] + ['SR'] + ['REQ/s_' + s for s in CONFIG.services]
    return {
        'DF_ALL': df,
        'LOAD_MAPS': maps,
        'RES_TIME': {
            'DF_ALL': df[cols_base_all + ["RES_TIME_" + s for s in CONFIG.services]],
            'DF_DISC': df_d[cols_base_disc + ["RES_TIME_" + s for s in CONFIG.services]]
        },
        'CPU': {
            'DF_ALL': df[cols_base_all + ["CPU_" + s for s in CONFIG.services]],
            'DF_DISC': df_d[cols_base_disc + ["CPU_" + s for s in CONFIG.services]]
        },
        'MEM': {
            'DF_ALL': df[cols_base_all + ["MEM_" + s for s in CONFIG.services]],
            'DF_DISC': df_d[cols_base_disc + ["MEM_" + s for s in CONFIG.services]]
        }
    }


def discovery(df_dict, dags_dir):
    print("----------CAUSAL DISCOVERY----------")

    for met in CONFIG.all_metrics:
        print("  -----DLINGAM WITH PRIOR FOR {}-----".format(met))

        df_discovery_met = df_dict[met]['DF_DISC']
        maps = {i: df_discovery_met.columns[i] for i in range(0, len(df_discovery_met.columns))}
        X = df_discovery_met.to_numpy(dtype=np.float64)

        prior = utils.get_generic_priorknorledge_mat(df_discovery_met.columns, CONFIG.services, CONFIG.path_wm,
                                                     num_load=len(CONFIG.loads), metrics=[met])
        qu.save_adjusted_dag(os.path.join(dags_dir, "dlingam_prior_" + met),
                             qu.dlingam_discovery(X, prior_knowledge=prior), maps)


def configuration_generation(df_dict, dags_dir, config_dir):
    print("----------CONFIGURATION GENERATION----------")

    for met in CONFIG.all_metrics:
        print("  -----DLINGAM WITH PRIOR FOR {}-----".format(met))

        qu.gen_configs_per_metric(df_dict[met]['DF_DISC'], df_dict['LOAD_MAPS'],
                                  os.path.join(dags_dir, "dlingam_prior_{}.dot".format(met)),
                                  os.path.join(config_dir, "dlingam_prior"), metrics=[met])


def show_metrics(df_dict, config_dir, path_mets):
    print("----------METRICS----------")
    mets_models = {}
    for met in CONFIG.all_metrics:
        mod = "dlingam_prior_" + met
        print("  -----DLINGAM WITH PRIOR FOR {}-----".format(met))
        mets_models[mod] = qu.calc_metrics(df_dict[met]['DF_ALL'],
                                           path_configs=os.path.join(config_dir, "dlingam_prior_" + met),
                                           metrics=[met])
        print("{} -> PREC: {:.2f} RECALL: {:.2f} MEAN DIST: {:.2f} MIN DIST: {} MAX DIST: {}"
              .format(met, mets_models[mod]['precision'], mets_models[mod]['recall'],
                      mets_models[mod]['mean_hamming_distance'], mets_models[mod]['min_hamming_distance'],
                      mets_models[mod]['max_hamming_distance']))
    with open(path_mets, 'w') as f:
        json.dump(mets_models, f)


def __main__(path_df, path_main_dir):
    path_dir_dags = os.path.join(path_main_dir, "dags")
    path_dir_configs = os.path.join(path_main_dir, "configs")
    path_metric_file = os.path.join(path_main_dir, "metrics.json")

    os.makedirs(path_dir_dags, exist_ok=True)
    os.makedirs(path_dir_configs, exist_ok=True)

    df_dict = data_preparation(path_df)
    discovery(df_dict, path_dir_dags)
    configuration_generation(df_dict, path_dir_dags, path_dir_configs)
    show_metrics(df_dict, path_dir_configs, path_metric_file)
