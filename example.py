import json
import os.path
import random
import shutil

import numpy as np
import pandas as pd
import data_preparation
import data_visualization
import utils
from causal_model_generator import build_model
from configuration_generator import generate_config
from enum import Enum

from performance_evaluator import calc_metrics

from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold

CURRENT_PATH = os.path.dirname(__file__)


# TODO: salvare il dataset originale nelle repliche
class System(Enum):
    MUBENCH = "mubench"
    SOCKSHOP = "sockshop"
    ONLINEBOUTIQUE = "onlineboutique"


def get_system_info(system):
    if system == System.MUBENCH:
        services = ['s' + str(i) for i in range(10)]
        mapping = {s: s for s in services}
        pods = services
        arch = utils.get_architecture_from_wm(os.path.join(CURRENT_PATH, "mubench/configs/workmodel.json"))
    else:
        with open(os.path.join(CURRENT_PATH, str(system.value), "architecture.json"), 'r') as f_arch:
            with open(os.path.join(CURRENT_PATH, str(system.value), "mapping_service_request.json"), 'r') as f_map:
                with open(os.path.join(CURRENT_PATH, str(system.value), "pods.txt"), 'r') as f_pods:
                    arch = json.load(f_arch)
                    mapping = json.load(f_map)
                    services = list(set(mapping))
                    pods = [p.replace("\n", "") for p in f_pods.readlines()]

    return services, mapping, pods, arch


def create_dataset(system, path_df=None, path_exp=None):
    services, mapping, pods, arch = get_system_info(system)
    if path_df is None:
        path_df = os.path.join(CURRENT_PATH, str(system.value), "data", str(system.value) + "_df.csv")
    if path_exp is None:
        path_exp = os.path.join(CURRENT_PATH, str(system.value), "data")
    df = data_preparation.read_experiments(path_exp, mapping, pods)
    df.to_csv(path_df, index=False)
    return df


def system_visualization(system, path_images=None):
    if path_images is None:
        path_images = os.path.join(CURRENT_PATH, str(system.value), "images")

    services, _, _, _ = get_system_info(system)
    path_df = os.path.join(CURRENT_PATH, str(system.value), "data", str(system.value) + "_df.csv")

    if not os.path.exists(path_df):
        df = create_dataset(system)
    else:
        df = pd.read_csv(path_df)

    if system == System.MUBENCH:
        df_disc = df[df['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
        data_visualization.nusers_vs_met(df_disc, path_images, services)
        data_visualization.loads_comparison(df_disc, path_images, services)
    else:
        data_visualization.nusers_vs_met(df, path_images, services, True)
        # data_visualization.loads_comparison(df, path_images, services, 1, True)


def system_workflow(system, path_work, dataset=None, generation_conf_FAST=False):
    services, mapping, pods, architecture = get_system_info(system)

    path_df = os.path.join(path_work, "df.csv")
    path_df_disc = os.path.join(path_work, "df_discovery.csv")
    path_thresholds = os.path.join(path_work, "thresholds.json")
    path_prior = os.path.join(path_work, "prior_knowledge")
    path_dag = os.path.join(path_work, "dag")
    path_configs = os.path.join(path_work, "generated_configs")

    if system == System.MUBENCH:
        ths_filtered = False
        model_limit = 30
        nuser_start = 2
    else:
        ths_filtered = True
        model_limit = 50
        nuser_start = 4

    if not os.path.exists(path_work):
        os.mkdir(path_work)
    if not os.path.exists(path_configs):
        os.mkdir(path_configs)

    print("------READING EXPERIMENTS------")
    if dataset is None:
        df = create_dataset(system, path_df=path_df, path_exp=os.path.join(str(system.value), "data"))
    else:
        df = dataset
    df_discovery, loads_mapping = utils.hot_encode_col_mapping(df, 'LOAD')
    df_discovery.to_csv(path_df_disc, index=False)

    thresholds = {}
    with open(path_thresholds, 'w') as f_ths:
        for ser in services:
            thresholds[ser] = utils.calc_thresholds(df, ser, filtered=ths_filtered)
        json.dump(thresholds, f_ths)

    print("------GENERATING PRIOR KNOWLEDGE------")
    pk = utils.get_generic_priorknorledge_mat(df_discovery.columns, services, mapping, architecture)
    utils.draw_prior_knwoledge_mat(pk, df_discovery.columns, path_prior)

    print("------CAUSAL DISCOVERY------")
    causal_model = build_model(df_discovery, path_dag, pk)

    print("------GENERATING CONFIGRATIONS-")

    for ser in services:
        for met in ['RES_TIME', 'CPU', 'MEM']:
            print("Searching configuration for service: {} for metric: {}".format(ser, met))
            generate_config(causal_model, df_discovery, ser,
                            os.path.join(path_configs, "{}_{}_{}.json".format("configs", met, ser)), loads_mapping,
                            metrics=[met], stability=0, nuser_limit=model_limit, show_comment=True,
                            FAST=generation_conf_FAST, ths_filtered=ths_filtered, nuser_start=nuser_start)


def random_predictor(system, path_work, dataset=None):
    gen_path = os.path.join(path_work, "generated_configs")
    os.makedirs(gen_path, exist_ok=True)

    if dataset is None:
        df = create_dataset(system, os.path.join(path_work, "df.csv"))
    else:
        df = dataset

    SRs = list(set(df['SR']))
    loads = list(set(df['LOAD']))
    if system == System.MUBENCH:
        model_limit = 30
        nstart = 2
    else:
        model_limit = 50
        nstart = 4

    services, _, _, _ = get_system_info(system)

    for ser in services:
        for met in ['RES_TIME', 'CPU', 'MEM']:
            with open(os.path.join(gen_path, 'configs_{}_{}.json'.format(met, ser)),
                      'w') as f_conf:
                json.dump({"nusers": [random.randint(nstart, model_limit)], "loads": [random.choice(loads)],
                           "spawn_rates": [random.choice(SRs)], "anomalous_metrics": [met]}, f_conf)


def mlp_predictor(system, path_work, dataset=None):
    services, mapping, pods, architecture = get_system_info(system)
    metrics = ['RES_TIME', 'CPU', 'MEM']

    path_df_train = os.path.join(path_work, "df_train.csv")
    path_thresholds = os.path.join(path_work, "thresholds.json")
    path_configs = os.path.join(path_work, "generated_configs")
    path_model = os.path.join(path_work, "model.h5")

    if system == System.MUBENCH:
        ths_filtered = False
        model_limit = 30
        nuser_start = 2
        path_read_df = "mubench/data/mubench_df.csv"
    else:
        ths_filtered = True
        model_limit = 50
        nuser_start = 4
        path_read_df = "sockshop/data/sockshop_df.csv"

    if not os.path.exists(path_work):
        os.mkdir(path_work)
    if not os.path.exists(path_configs):
        os.mkdir(path_configs)

    print("------READING EXPERIMENTS------")
    if dataset is None:
        df = pd.read_csv(path_read_df)
    else:
        df = dataset

    df_train, loads_mapping = utils.hot_encode_col_mapping(df, 'LOAD')
    # df_train = df_train[df_train['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
    input_col = ['NUSER'] + ['LOAD_' + str(i) for i in range(len(loads_mapping.keys()))] + ['SR']
    output_col = [met + "_" + ser for met in metrics for ser in services]
    df_train = df_train[input_col + output_col]
    # df_train[output_col] = (df_train[output_col] - df_train[output_col].min()) / (
    #            df_train[output_col].max() - df_train[output_col].min())
    df_train.to_csv(path_df_train, index=False)

    X_train = df_train[input_col].values
    Y_train = df_train[output_col].values

    loads = loads_mapping.keys()
    spawn_rates = list(set(df['SR']))

    thresholds = {}
    with open(path_thresholds, 'w') as f_ths:
        for ser in services:
            thresholds[ser] = utils.calc_thresholds(df_train, ser, filtered=ths_filtered)
        json.dump(thresholds, f_ths)

    print("------DEFINING MLP------")
    model = Sequential()
    model.add(Input(shape=(5,), dtype=int))  # settare interi
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=len(output_col), activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    print("------TRAIN------")
    kfold = KFold(n_splits=5, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train)):
        print(f"Fold {fold + 1}:")

        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = Y_train[train_index], Y_train[val_index]

        # Addestramento del modello su X_train_fold, y_train_fold
        model.fit(X_train_fold, y_train_fold, epochs=300, batch_size=32, verbose=0)

        # Valutazione del modello su X_val_fold, y_val_fold
        loss, accuracy = model.evaluate(X_val_fold, y_val_fold)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    model.save(path_model)

    print("------GENERATING CONFIGRATIONS-")

    def search_config():
        for n in range(nuser_start, model_limit + 1):
            print(n)
            for load in loads:
                for sr in spawn_rates:

                    input_data = np.zeros(len(input_col), )
                    input_data[input_col.index('NUSER')] = n
                    input_data[input_col.index('SR')] = sr

                    for li in range(len(loads)):
                        input_data[input_col.index('LOAD_{}'.format(li))] = loads_mapping[load][li]

                    target_pred = model.predict(np.array([input_data]))[:, [output_col.index(target_col)]]

                    if target_pred > thresholds[ser][met]:
                        print("CONFIGURAZIONE TROVATA")
                        with open(os.path.join(path_configs, "{}_{}_{}.json".format("configs", met, ser)),
                                  'w') as f_out:
                            json.dump({"nusers": [n], "loads": [load], "spawn_rates": [sr],
                                       "anomalous_metrics": [met]}, f_out)
                        return
        print("NON TROVATA")

    for ser in services:
        for met in metrics:
            target_col = met + "_" + ser
            print("Searching configuration for service: {} for metric: {}".format(ser, met))

            search_config()


def system_performance_evaluation(system, from_experiments, path_work=None, sensibility=0., df_exps=None):
    if not from_experiments and df_exps is None:
        print("ERRORE: NON È STATI PASSATO IL DATASET EXPS")  # TODO: lanciare eccezione
        quit()

    if system == System.MUBENCH:
        model_limit = 30
        if path_work is None:
            path_work = os.path.join(CURRENT_PATH, "mubench")
    else:
        model_limit = 50
        if path_work is None:
            if system == System.SOCKSHOP:
                path_work = os.path.join(CURRENT_PATH, "sockshop")
            else:
                path_work = os.path.join(CURRENT_PATH, "onlineboutique")

    path_df = os.path.join(path_work, "df.csv")
    if sensibility > 0:
        path_metrics = os.path.join(path_work, "metrics{}.json".format(sensibility))
    else:
        path_metrics = os.path.join(path_work, "metrics.json")
    path_configs = os.path.join(path_work, "generated_configs")
    path_run_configs = os.path.join(path_work, "run_configs")

    services, mapping, _, _ = get_system_info(system)

    def get_config_from_experiments(ser, met, df_experiments):
        name_config = "configs_{}_{}".format(met, ser)
        path_config = os.path.join(path_configs, name_config + ".json")
        if os.path.isfile(path_config):
            with open(path_config, 'r') as f_c:
                c = json.load(f_c)
                exp_dir = os.path.join(path_run_configs, name_config,
                                       "experiments_sr_" + str(c['spawn_rates'][0]),
                                       'users_' + str(c['nusers'][0]), c['loads'][0])
                return c, utils.get_experiment_value(exp_dir, ser, mapping[ser], met)
        else:
            return None, None

    def get_config_from_dataset(ser, met, df_experiments):
        path_conf = os.path.join(path_configs, "configs_{}_{}.json".format(met, ser))
        if os.path.isfile(path_conf):
            with open(path_conf, 'r') as f_c:
                config = json.load(f_c)
                df_config = df_experiments[(df_experiments['NUSER'] == config['nusers'][0]) &
                                           (df_experiments['LOAD'] == config['loads'][0]) &
                                           (df_experiments['SR'] == config['spawn_rates'][0])]

                if len(df_config) == 0:
                    print("MANCA: ({},{},{})".format(config['nusers'][0], config['loads'][0],
                                                     config['spawn_rates'][0]))
                return config, df_config[met + "_" + ser].mean()
        else:
            return None, None

    if from_experiments:
        metrics = calc_metrics(path_df, get_config_from_experiments, services, model_limit,
                               ths_filtered=(system != System.MUBENCH),
                               sensibility=sensibility)
    else:
        metrics = calc_metrics(path_df, get_config_from_dataset, services, model_limit,
                               ths_filtered=(system != System.MUBENCH),
                               sensibility=sensibility, df_exps=df_exps)

    with open(path_metrics, 'w') as f_metrics:
        json.dump(metrics, f_metrics)


def system_mean_performance(system, from_experiments, reps, name_sub_dir="work_rep", path_works=None, sensibility=0.,
                            df_exps=None):
    if path_works is None:
        if system == System.MUBENCH:
            path_works = os.path.join(CURRENT_PATH, "mubench")
        elif system == System.SOCKSHOP:
            path_works = os.path.join(CURRENT_PATH, "sockshop/works")
        else:
            path_works = os.path.join(CURRENT_PATH, "onlineboutique/works")

    if sensibility > 0:
        path_mean_metrics = os.path.join(path_works, 'avg_metrics_{}.json'.format(sensibility))
    else:
        path_mean_metrics = os.path.join(path_works, 'avg_metrics.json')

    metrics = ['RES_TIME', 'CPU', 'MEM']
    mean_metrics = {}
    metrics_reps = {}

    for met in metrics:
        metrics_reps[met] = {
            'precision': [0.] * len(reps),
            'recall': [0.] * len(reps),
            'mhd_pos': [0.] * len(reps),
            'mhd_false': [0.] * len(reps),
            'true_positive': [0.] * len(reps),
            'true_negative': [0.] * len(reps),
            'false_positive': [0.] * len(reps),
            'false_negative': [0.] * len(reps)
        }
        mean_metrics[met] = {}

    for k, rep in enumerate(reps):
        path_rep = os.path.join(path_works, name_sub_dir + str(rep))
        if sensibility > 0:
            path_rep_metrics = os.path.join(path_rep, "metrics{}.json".format(sensibility))
        else:
            path_rep_metrics = os.path.join(path_rep, "metrics.json")

        system_performance_evaluation(system, from_experiments, path_rep, sensibility, df_exps=df_exps)

        with open(path_rep_metrics, 'r') as f_mets:
            mets = json.load(f_mets)
            for met in metrics:
                for key, val in mets[met].items():
                    metrics_reps[met][key][k] = val

    for met in metrics:
        for key, val in metrics_reps[met].items():
            if "mhd" not in key:
                mean_metrics[met][key + '_mean'] = round(np.mean(val), 3)
                mean_metrics[met][key + '_std'] = round(np.std(val), 3)
            else:
                filtered_vals = [v for v in val if v >= 0]
                if len(filtered_vals) > 0:
                    mean_metrics[met][key + '_mean'] = round(np.mean(filtered_vals), 3)
                    mean_metrics[met][key + '_std'] = round(np.std(filtered_vals), 3)
                else:
                    mean_metrics[met][key + '_mean'] = -1
                    mean_metrics[met][key + '_std'] = -1

    with open(path_mean_metrics, 'w') as f_metrics:
        json.dump(mean_metrics, f_metrics)


# TODO: rifare con whereis di pandas
def anomalies_filter(df_in, services_in, mets, ths_filtered=False):
    cancel_index = []

    for i in df_in.index:
        def a():
            for ser in services_in:
                ths = utils.calc_thresholds(df_in, ser, filtered=ths_filtered)
                for met in mets:
                    if df_in.loc[i, met + "_" + ser] > ths[met]:
                        cancel_index.append(i)
                        return

        a()

    return df_in.drop(index=cancel_index).reset_index(drop=True)


create_dataset(System.ONLINEBOUTIQUE,"onlineboutique/compact_configs_run_old/df.csv", "onlineboutique/compact_configs_run_old")


quit()
create_dataset(System.ONLINEBOUTIQUE,"onlineboutique/data/onlineboutique_df.csv", "onlineboutique/data")

count = 0
count_tot = 0
count_no = 0
services, mapping, pods, arch = get_system_info(System.ONLINEBOUTIQUE)

df = pd.read_csv("onlineboutique/data/onlineboutique_df.csv")
for ser in services:
    ths = utils.calc_thresholds(df, ser, True)
    for met in ['RES_TIME', 'MEM', 'CPU']:
        n_min = 51
        count_tot += 1
        for l in list(set(list(df['LOAD']))):
            for sr in list(set(list(df['SR']))):

                users = list(set(list(df['NUSER'])))
                users.sort()
                for n in users:
                    if df[(df['NUSER'] == n) & (df['LOAD'] == l) & (df['SR'] == sr)][met + "_" + ser].mean() > ths[met]:
                        n_min = min(n_min, n)
                        break
        print("{}_{} -> {}".format(met, ser, n_min))
        if 4 < n_min < 51:
            count += 1
        if n_min == 51:
            count_no += 1

print("{}+{}/{}".format(count,count_no,count_tot))
quit()

for rep in range(20):
    system_workflow(System.ONLINEBOUTIQUE, "onlineboutique/NO_ANOMALY/works_causal/rep" + str(rep),
                    pd.read_csv("onlineboutique/NO_ANOMALY/df_onlineboutique_filtered.csv"))

for rep in range(20):
    mlp_predictor(System.ONLINEBOUTIQUE, "onlineboutique/NO_ANOMALY/works_mlp/rep" + str(rep),
                  pd.read_csv("onlineboutique/NO_ANOMALY/df_onlineboutique_filtered.csv"))
