import json
import os.path
import shutil

import lingam
import numpy as np
import pandas as pd
from dowhy import gcm

from configuration_generator import generate_config
import utils
import data_preparation


def build_model(df, path_dag, prior_knowledge=None):
    X = df.to_numpy(dtype=np.float64)
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)
    adj_mat = utils.threshold_matrix(np.transpose(model.adjacency_matrix_), 0)

    causal_graph = utils.adjmat2dot(adj_mat, df.columns)
    utils.save_dot(causal_graph, path_dag)
    causal_model = gcm.StructuralCausalModel(causal_graph)
    gcm.auto.assign_causal_mechanisms(causal_model, df, quality=gcm.auto.AssignmentQuality.GOOD)
    gcm.fit(causal_model, df)
    # TODO: salvare causal model
    return causal_model


def system_example(path_exp, path, architecture, pods, mapping, generation_conf_FAST=False):
    path_df = os.path.join(path, "df.csv")
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

    thresholds = {}
    with open(path_thresholds, 'w') as f_ths:
        for ser in services:
            thresholds[ser] = utils.calc_thresholds(df, ser)
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
                            metrics=[met], stability=0, nuser_limit=50, show_comment=True, FAST=generation_conf_FAST)
        print("Searching configuration for service: {} for all metrics".format(ser))
        generate_config(causal_model, df_discovery, ser,
                        os.path.join(path_configs, "{}_{}_{}.json".format("configs", "all", ser)), loads_mapping,
                        metrics=None, stability=0, nuser_limit=50, show_comment=True, FAST=generation_conf_FAST)


def muBench_example(generation_conf_FAST=False):
    def get_arch_from_wm(path_wm):
        arch = {}
        with open(path_wm) as f_wm:
            wm = json.load(f_wm)
            for k, v in wm.items():
                if len(v['external_services']) > 0:
                    arch[k] = []
                    for eser in v['external_services']:
                        for ser in eser['services']:
                            arch[k].append(ser)
        return arch

    services = ['s' + str(i) for i in range(10)]
    system_example("mubench/data", "mubench/work", get_arch_from_wm("mubench/configs/workmodel.json"), services,
                   {s: s for s in services}, generation_conf_FAST)


def sockshop_example(generation_conf_FAST=False):
    with open("sockshop/architecture.json", 'r') as f_arch:
        with open("sockshop/pods.txt", 'r') as f_pods:
            with open("sockshop/mapping_service_request.json", "r") as f_map:
                system_example("sockshop/data", "sockshop/work", json.load(f_arch),
                               [p.replace("\n", "") for p in f_pods.readlines()], json.load(f_map),
                               generation_conf_FAST)


def trainticket_example(generation_conf_FAST=False):
    with open("trainticket/architecture.json", 'r') as f_arch:
        with open("trainticket/pods.txt", 'r') as f_pods:
            with open("trainticket/mapping_service_request.json", "r") as f_map:
                system_example("trainticket/data", "trainticket/work", json.load(f_arch),
                               [p.replace("\n", "") for p in f_pods.readlines()], json.load(f_map),
                               generation_conf_FAST)


muBench_example(True)
