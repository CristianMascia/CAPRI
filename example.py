import json
import os.path
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
        with open("sockshop/architecture.json", 'r') as f_arch:
            with open(os.path.join(str(system.value), "mapping_service_request.json"), 'r') as f_map:
                with open(os.path.join(str(system.value), "mapping_service_request.json"), 'r') as f_pods:
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


def system_performance_evaluation(system, path_work):
    path_df = os.path.join(path_work, "df.csv")
    path_metrics = os.path.join(path_work, "metrics.json")
    path_configs = os.path.join(path_work, "generated_configs")
    path_run_configs = os.path.join(path_work, "run_configs")

    services, mapping, _, _ = get_system_info(system)
    metrics = calc_metrics(path_df, path_configs, path_run_configs, services, mapping)

    with open(path_metrics, 'w') as f_metrics:
        json.dump(metrics, f_metrics)
