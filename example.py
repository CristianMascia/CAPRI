import json
import os.path
import random

import numpy as np
import pandas as pd
import data_preparation
import data_visualization
import utils
from causal_model_generator import build_model
from configuration_generator import generate_config
from enum import Enum

from performance_evaluator import calc_metrics


class System(Enum):
    MUBENCH = "mubench"
    SOCKSHOP = "sockshop"
    TRAINTICKET = "trainticket"


def get_system_info(system):
    if system == System.MUBENCH:
        services = ['s' + str(i) for i in range(10)]
        mapping = {s: s for s in services}
        pods = services
        arch = utils.get_architecture_from_wm("mubench/configs/workmodel.json")
    else:
        with open(os.path.join(str(system.value), "architecture.json"), 'r') as f_arch:
            with open(os.path.join(str(system.value), "mapping_service_request.json"), 'r') as f_map:
                with open(os.path.join(str(system.value), "pods.txt"), 'r') as f_pods:
                    arch = json.load(f_arch)
                    mapping = json.load(f_map)
                    services = list(set(mapping))
                    pods = [p.replace("\n", "") for p in f_pods.readlines()]

    return services, mapping, pods, arch


def create_dataset(system, path_df=None, path_exp=None):
    services, mapping, pods, arch = get_system_info(system)
    if path_df is None:
        path_df = os.path.join(str(system.value), "data", str(system.value) + "_df.csv")
    if path_exp is None:
        path_exp = os.path.join(str(system.value), "data")
    df = data_preparation.read_experiments(path_exp, mapping, pods)
    df.to_csv(path_df, index=False)
    return df


def system_visualization(system, path_png=None):
    if path_png is None:
        path_png = os.path.join(str(system.value), "png")

    services, _, _, _ = get_system_info(system)
    path_df = os.path.join(str(system.value), "data", str(system.value) + "_df.csv")

    if not os.path.exists(path_df):
        df = create_dataset(system)
    else:
        df = pd.read_csv(path_df)

    if system == System.MUBENCH:
        df_disc = df[df['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
        data_visualization.___main__(df_disc, path_png, services)
    else:
        data_visualization.___main__(df, path_png, services, True)


def system_workflow(system, path_work, generation_conf_FAST=False):
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
    df = create_dataset(system, path_df=path_df, path_exp=os.path.join(str(system.value), "data"))
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


def random_predictor(system, path_work):
    gen_path = os.path.join(path_work, "generated_configs")
    os.makedirs(gen_path, exist_ok=True)

    SRs = [1, 5, 10]
    if system == System.MUBENCH:
        model_limit = 30
        nstart = 2
        loads = ['uniform', 'randomly_balanced', 'unbalanced_one']
    else:
        model_limit = 50
        nstart = 4
        if system == System.SOCKSHOP:
            loads = ['normal', 'stress_cart', 'stress_shop']
        else:
            loads = ['normal', 'stress_booking', 'stress_cancel']

    services, _, _, _ = get_system_info(system)

    for ser in services:
        for met in ['RES_TIME', 'CPU', 'MEM']:
            with open(os.path.join(gen_path, 'configs_{}_{}.json'.format(met, ser)),
                      'w') as f_conf:
                json.dump({"nusers": [random.randint(nstart, model_limit + 1)], "loads": [random.choice(loads)],
                           "spawn_rates": [random.choice(SRs)], "anomalous_metrics": [met]}, f_conf)


def system_performance_evaluation(system, path_work, sensibility=0.):
    path_df = os.path.join(path_work, "df.csv")
    path_metrics = os.path.join(path_work, "metrics{}.json".format(sensibility))
    path_configs = os.path.join(path_work, "generated_configs")
    path_run_configs = os.path.join(path_work, "run_configs")

    services, mapping, _, _ = get_system_info(system)
    metrics = calc_metrics(path_df, path_configs, path_run_configs, services, mapping,
                           ths_filtered=(system != System.MUBENCH), sensibility=sensibility)

    with open(path_metrics, 'w') as f_metrics:
        json.dump(metrics, f_metrics)


def system_mean_performance(system, path_works, reps, sensibility=0.):
    path_mean_metrics = os.path.join(path_works, 'avg_metrics_{}.json'.format(sensibility))
    metrics = ['RES_TIME', 'CPU', 'MEM']
    mean_metrics = {}
    metrics_reps = {}

    for met in metrics:
        metrics_reps[met] = {
            'precision': [0.] * len(reps),
            'recall': [0.] * len(reps),
            'mhd': [0.] * len(reps),
        }
        mean_metrics[met] = {
            'precision_mean': 0.,
            'recall_mean': 0.,
            'precision_std': 0.,
            'recall_std': 0.,
            "mhd_mean": 0.,
            "mhd_std": 0.
        }

    for k, rep in enumerate(reps):
        path_rep = os.path.join(path_works, "work_rep" + str(rep))
        path_rep_metrics = os.path.join(path_rep, "metrics{}.json".format(sensibility))

        system_performance_evaluation(system, path_rep, sensibility)

        with open(path_rep_metrics, 'r') as f_mets:
            mets = json.load(f_mets)
            for met in metrics:
                metrics_reps[met]['precision'][k] = mets[met]['precision']
                metrics_reps[met]['recall'][k] = mets[met]['recall']
                metrics_reps[met]['mhd'][k] = mets[met]['mhd']

    for met in metrics:
        mean_metrics[met]['precision_mean'] = round(np.mean(metrics_reps[met]['precision']), 3)
        mean_metrics[met]['precision_std'] = round(np.std(metrics_reps[met]['precision']), 3)
        mean_metrics[met]['recall_mean'] = round(np.mean(metrics_reps[met]['recall']), 3)
        mean_metrics[met]['recall_std'] = round(np.std(metrics_reps[met]['recall']), 3)
        mean_metrics[met]['mhd_mean'] = round(np.mean(metrics_reps[met]['mhd']), 3)
        mean_metrics[met]['mhd_std'] = round(np.std(metrics_reps[met]['mhd']), 3)

    with open(path_mean_metrics, 'w') as f_metrics:
        json.dump(mean_metrics, f_metrics)
