import json
import os
import numpy as np
import pandas as pd
import questions_utils as qu
import utils

######## How does performance vary if we consider one model per performance metric or one model for all metrics?

dags_dir = "question2/dags/"
config_dir = "question2/configs/"
path_mets = "question2/metrics.json"

os.makedirs(dags_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)


def data_preparation():
    print("----------DATA PREPARATION----------")
    df = pd.read_csv("../mubench/data/mubench_df.csv")
    df_d, maps = utils.get_discovery_dataset(df)
    cols_base_all = ['NUSER', 'LOAD', 'SR'] + ['REQ/s_' + s for s in qu.services]
    cols_base_disc = ['NUSER'] + ['LOAD_' + str(i) for i in range(3)] + ['SR'] + ['REQ/s_' + s for s in uq.services]
    return {
        'DF_ALL': df,
        'LOAD_MAPS': maps,
        'RES_TIME': {
            'DF_ALL': df[cols_base_all + ["RES_TIME_" + s for s in uq.services]],
            'DF_DISC': df_d[cols_base_disc + ["RES_TIME_" + s for s in uq.services]]
        },
        'CPU': {
            'DF_ALL': df[cols_base_all + ["CPU_" + s for s in uq.services]],
            'DF_DISC': df_d[cols_base_disc + ["CPU_" + s for s in uq.services]]
        },
        'MEM': {
            'DF_ALL': df[cols_base_all + ["MEM_" + s for s in uq.services]],
            'DF_DISC': df_d[cols_base_disc + ["MEM_" + s for s in uq.services]]
        }
    }


def discovery(df_dict):
    print("----------CAUSAL DISCOVERY----------")

    for met in qu.all_metrics:
        print("  -----DLINGAM WITH PRIOR FOR {}-----".format(met))

        df_discovery_met = df_dict[met]['DF_DISC']
        maps = {i: df_discovery_met.columns[i] for i in range(0, len(df_discovery_met.columns))}
        X = df_discovery_met.to_numpy(dtype=np.float64)

        prior = utils.get_generic_priorknorledge_mat(df_discovery_met.columns, qu.services, qu.path_wm,
                                                     num_load=len(qu.loads), metrics=[met])
        qu.save_adjusted_dag(dags_dir + "dlingam_prior_" + met, qu.dlingam_discovery(X, prior_knowledge=prior), maps)


def configuration_generation(df_dict):
    print("----------CONFIGURATION GENERATION----------")

    for met in qu.all_metrics:
        print("  -----DLINGAM WITH PRIOR FOR {}-----".format(met))

        qu.gen_configs_per_metric(df_dict[met]['DF_DISC'], df_dict['LOAD_MAPS'],
                                  dags_dir + "dlingam_prior_" + met + ".dot",
                                  config_dir + "dlingam_prior", metrics=[met])


def show_metrics(df_dict):
    print("----------METRICS----------")
    mets_models = {}
    for met in qu.all_metrics:
        mod = "dlingam_prior_" + met
        print("  -----DLINGAM WITH PRIOR FOR {}-----".format(met))
        mets_models[mod] = qu.calc_metrics(df_dict[met]['DF_ALL'], path_configs=config_dir + "dlingam_prior_" + met,
                                           metrics=[met])
        print("PREC: {:.2f} RECALL: {:.2f}".format(mets_models[mod]['precision'], mets_models[mod]['recall']))

    with open(path_mets, 'w') as f:
        json.dump(mets_models, f)


def __main__():
    df_dict = data_preparation()
    discovery(df_dict)
    configuration_generation(df_dict)
    show_metrics(df_dict)


__main__()
