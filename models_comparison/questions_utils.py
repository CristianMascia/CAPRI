import json
import os
import lingam
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from dowhy import gcm
import configuration_generator as conf_gen
import utils

services = ['s' + str(i) for i in range(10)]
loads = ['uniform', 'randomly_balanced', 'unbalanced_one']
SRs = [1, 5, 10]
treatments = ['NUSER', 'LOAD_0', 'LOAD_1', 'LOAD_2', 'SR']
all_metrics = ['RES_TIME', 'CPU', 'MEM']
mubench_path = "../mubench"
path_df = mubench_path + "/data/mubench_df.csv"
path_wm = mubench_path + "/configs/workmodel.json"
path_minimal_configs_anomaly = mubench_path + "/minimal_configs_anomaly.json"
model_limit = 30


# remove or invert in-edge for threatment nodes
def adjust_dag(dag):
    # remove self-loop edges
    for n in dag.nodes:
        if dag.has_edge(n, n):
            dag.remove_edge(n, n)

    for t in treatments:
        for n in [i for i in dag.predecessors(t)]:
            dag.remove_edge(n, t)
            if n not in treatments:
                dag.add_edge(t, n)
    return dag


def save_dag(path, adj, maps):
    np.save(path, adj)
    utils.save_dot(utils.adjmat2dot_map(adj, maps), path)


def save_adjusted_dag(path, adj, maps):
    dag = adjust_dag(utils.adjmat2dot_map(adj, maps))
    np.save(path, nx.adjacency_matrix(dag, [maps[key] for key in sorted(maps.keys())], dtype=int))
    utils.save_dot(dag, path)


def dlingam_discovery(X, th=0, prior_knowledge=None):
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)
    return utils.threshold_matrix(np.transpose(model.adjacency_matrix_), th=th)


def dagma_lin_discovery(X, th=0):
    torch.set_default_dtype(torch.float64)
    model_dagma = DagmaLinear(loss_type='l2')
    return utils.threshold_matrix(model_dagma.fit(X, lambda1=0.02), th=th)


def dagma_mlp_discovery(X, th=0):
    torch.set_default_dtype(torch.float64)
    eq_model = DagmaMLP(dims=[X.shape[1], 10, 1], bias=True, dtype=torch.double)
    model = DagmaNonlinear(eq_model, dtype=torch.double)
    return utils.threshold_matrix(model.fit(X, lambda1=0.02, lambda2=0.005), th=th)


def gen_configs_per_metric(df_discovery, loads_mapping, path_cgraph, path_configs, metrics=None):
    if metrics is None:
        metrics = all_metrics

    cg = nx.DiGraph(nx.nx_pydot.read_dot(path_cgraph))
    if cg.has_node("\\n"):  # TODO: BUG
        cg.remove_node("\\n")

    causal_model = gcm.StructuralCausalModel(cg)
    gcm.auto.assign_causal_mechanisms(causal_model, df_discovery, quality=gcm.auto.AssignmentQuality.GOOD)
    gcm.fit(causal_model, df_discovery)

    for ser in services:
        for met in metrics:
            conf_gen.generate_config(causal_model, df_discovery, ser,
                                     "{}_{}_{}.json".format(path_configs, met, ser),
                                     loads_mapping,
                                     metrics=[met],
                                     stability=0, nuser_limit=model_limit)


def create_min_anomalies_file(df, path, metrics=None):
    if metrics is None:
        metrics = all_metrics

    ns = list(set(df['NUSER']))

    out_j = {}
    for ser in services:
        out_j[ser] = {}
        ths = utils.calc_thresholds(df, ser)
        for met in metrics:
            for n in ns:
                icomb = []
                for l in loads:
                    for sr in SRs:
                        if df[(df['NUSER'] == n) & (df['LOAD'] == l) & (df['SR'] == sr)][met + '_' + ser].mean() > ths[
                            met]:
                            icomb.append({'NUSER': n, 'LOAD': l, 'SR': sr})

                # out_j[ser][met] = {"NUSER": n, 'LOAD': l, 'SR': sr}
                if len(icomb) > 0:
                    out_j[ser][met] = icomb
                    break

    with open(path, 'w') as f_out:
        json.dump(out_j, f_out)


def calc_metrics(df, path_configs, metrics=None):
    if metrics is None:
        metrics = all_metrics

    def calc_hamming_distance(conf_real, conf_pred):
        # a = abs(conf_real['NUSER'] - conf_pred['nusers'][0])
        # b = 0 if conf_real['LOAD'] == conf_pred['loads'][0] else 1
        # c = 0 if conf_real['SR'] == conf_pred['spawn_rates'][0] else 1
        # d = (conf_pred['spawn_rates'][0])
        # print("a:{} b:{} c:{} d:{}".format(a, b, c, d))

        return abs(conf_real['NUSER'] - conf_pred['nusers'][0]) + (
            0 if conf_real['LOAD'] == conf_pred['loads'][0] else 1) + (
            0 if conf_real['SR'] == conf_pred['spawn_rates'][0] else 1)

    if not os.path.exists(path_minimal_configs_anomaly):
        create_min_anomalies_file(pd.read_csv(path_df), path_minimal_configs_anomaly)

    with open(path_minimal_configs_anomaly, 'r') as f_minimal_conf:
        minimal_confs = json.load(f_minimal_conf)

        true_positive = 0
        false_negative = 0
        false_positive = 0
        dists = []

        for ser in services:

            path_config = path_configs + "_" + ser + ".json"
            ths = utils.calc_thresholds(df, ser)

            if os.path.exists(path_config):
                with open(path_config, 'r') as f_c:
                    config = json.load(f_c)

                    df_config = df[(df['NUSER'] == config['nusers'][0]) &
                                   (df['LOAD'] == config['loads'][0]) &
                                   (df['SR'] == config['spawn_rates'][0])]

                    # X - Positive
                    for met in config['anomalous_metrics']:
                        if df_config[met + "_" + ser].mean() > ths[met]:  # True - Positive
                            true_positive += 1
                            dists_c = np.zeros(len(minimal_confs[ser][met]), dtype=int)
                            for i, mc in enumerate(minimal_confs[ser][met]):
                                dists_c[i] = calc_hamming_distance(mc, config)
                                # print("R: ({},{},{}) P: ({},{},{}) -> {}".format(mc['NUSER'], mc['LOAD'], mc['SR'],
                                #                                                 config['nusers'][0],
                                #                                                 config['loads'][0],
                                #                                                 config['spawn_rates'][0],
                                #                                                 calc_hamming_distance(mc, config)))

                            dists.append(min(dists_c))
                        else:  # False - Positive
                            false_positive += 1

            else:  # X - Negative
                false_negative = 0
                # check there is anomalies for service ser
                # if check_anomalies(df, ser, metrics=metrics):  # False - Negative
                if len(minimal_confs[ser]) > 0:  # False - Negative
                    false_negative += 1

        precision = 0
        recall = 0

        if true_positive > 0:
            precision = true_positive / (true_positive + false_positive)

            if false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 1

        return {'precision': precision, 'recall': recall, 'mean_hamming_distance': np.mean(dists)}


m = calc_metrics(pd.read_csv(path_df), "question1/configs/dlingam_prior_RES_TIME", metrics=['RES_TIME'])
print(m)
