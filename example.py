import json
import os.path

import pandas as pd

import data_preparation
import data_visualization
import utils
from causal_model_generator import build_model
from configuration_generator import generate_config


def system_workflow(path_exp, path, architecture, pods, mapping, ths_filtered=False, generation_conf_FAST=False):
    path_df = os.path.join(path, "df.csv")
    path_df_disc = os.path.join(path, "df_discovery.csv")
    path_thresholds = os.path.join(path, "thresholds.json")
    path_prior = os.path.join(path, "prior_knowledge")
    path_dag = os.path.join(path, "dag")
    path_configs = os.path.join(path, "generated_configs")

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path_configs):
        os.mkdir(path_configs)

    services = list(mapping.keys())

    print("------READING EXPERIMENTS------")
    df = data_preparation.read_experiments(path_exp, mapping, pods, data_preparation.rename_startwith)
    df.to_csv(path_df, index=False)
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
                            metrics=[met], stability=0, nuser_limit=50, show_comment=True, FAST=generation_conf_FAST,
                            ths_filtered=ths_filtered)


def muBench_workflow(ths_filtered=False, generation_conf_FAST=False):
    path_system = "mubench"
    path_exp = os.path.join(path_system, "data")
    path_work = os.path.join(path_system, "work")
    path_wm = os.path.join(path_system, "configs", "workmodel.json")
    services = ['s' + str(i) for i in range(10)]
    system_workflow(path_exp, path_work, utils.get_architecture_from_wm(path_wm), services, {s: s for s in services},
                    ths_filtered=ths_filtered, generation_conf_FAST=generation_conf_FAST)


def sockshop_workflow(ths_filtered=False, generation_conf_FAST=False):
    path_system = "sockshop"
    path_exp = os.path.join(path_system, "data")
    path_work = os.path.join(path_system, "work")
    path_arch = os.path.join(path_system, "architecture.json")
    path_pods = os.path.join(path_system, "pods.txt")
    path_mapping = os.path.join(path_system, "mapping_service_request.json")

    with open(path_arch, 'r') as f_arch:
        with open(path_pods, 'r') as f_pods:
            pods = [p.replace("\n", "") for p in f_pods.readlines()]
            with open(path_mapping, "r") as f_map:
                system_workflow(path_exp, path_work, json.load(f_arch), pods, json.load(f_map),
                                ths_filtered=ths_filtered, generation_conf_FAST=generation_conf_FAST)


def trainticket_workflow(ths_filtered=False, generation_conf_FAST=False):
    path_system = "trainticket"
    path_exp = os.path.join(path_system, "data")
    path_work = os.path.join(path_system, "work")
    path_arch = os.path.join(path_system, "architecture.json")
    path_pods = os.path.join(path_system, "pods.txt")
    path_mapping = os.path.join(path_system, "mapping_service_request.json")
    with open(path_arch, 'r') as f_arch:
        with open(path_pods, 'r') as f_pods:
            pods = [p.replace("\n", "") for p in f_pods.readlines()]
            with open(path_mapping, "r") as f_map:
                system_workflow(path_exp, path_work, json.load(f_arch), pods, json.load(f_map),
                                ths_filtered=ths_filtered, generation_conf_FAST=generation_conf_FAST)


def mubench_visualization():
    path_system = "mubench"
    path_df = os.path.join(path_system, "data", "mubench_df.csv")
    path_png = os.path.join(path_system, "png")

    df = pd.read_csv(path_df)
    df_disc = df[df['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
    data_visualization.___main__(df_disc, path_png, ['s' + str(i) for i in range(10)])


def sockshop_visualization():
    path_system = "sockshop"
    path_mapping = os.path.join(path_system, "mapping_service_request.json")
    path_data = os.path.join(path_system, "data")
    path_df = os.path.join(path_data, "sockshop_df.csv")
    path_png = os.path.join(path_system, "png")

    with open(path_mapping, 'r') as f_map:
        data_visualization.___main__(pd.read_csv(path_df), path_png, list(set(json.load(f_map).keys())), True)


def trainticket_visualization():
    path_system = "trainticket"
    path_mapping = os.path.join(path_system, "mapping_service_request.json")
    path_data = os.path.join(path_system, "data")
    path_df = os.path.join(path_data, "trainticket_df.csv")
    path_png = os.path.join(path_system, "png")

    with open(path_mapping, 'r') as f_map:
        data_visualization.___main__(pd.read_csv(path_df), path_png, list(set(json.load(f_map).keys())))

