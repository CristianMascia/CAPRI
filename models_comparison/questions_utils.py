import json
import os
import lingam
import networkx as nx
import numpy as np
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
path_wm = "../mubench/configs/workmodel.json"
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


def check_anomalies(df, ser, metrics=None):
    if metrics is None:
        metrics = all_metrics

    ns = list(set(df['NUSER']))
    ns.sort(reverse=True)

    for n in ns:
        for l in loads:
            for sr in SRs:
                ths = utils.calc_thresholds(df, ser)
                for met in metrics:
                    if df[(df['NUSER'] == n) & (df['LOAD'] == l) & (df['SR'] == sr)][met + '_' + ser].mean() > ths[met]:
                        return True
    return False


def calc_metrics(df, path_configs, metrics=None):
    if metrics is None:
        metrics = all_metrics

    true_positive = 0
    false_negative = 0
    false_positive = 0

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
                    else:  # False - Positive
                        false_positive += 1
        else:  # X - Negative
            # check there is anomalies for service ser
            if check_anomalies(df, ser, metrics=metrics):  # False - Negative
                false_negative += 1

    precision = 0
    recall = 0

    if true_positive > 0:
        precision = true_positive / (true_positive + false_positive)

        if false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 1

    return {'precision': precision, 'recall': recall}
