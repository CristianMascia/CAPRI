import fileinput
import json
import os
import random

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
import shutil
import subprocess
import CONFIG


# remove or invert in-edge for treatment nodes
def adjust_dag(dag):
    # remove self-loop edges
    for n in dag.nodes:
        if dag.has_edge(n, n):
            dag.remove_edge(n, n)

    for t in CONFIG.treatments:
        for n in [i for i in dag.predecessors(t)]:
            dag.remove_edge(n, t)
            if n not in CONFIG.treatments:
                dag.add_edge(t, n)

    # remove cycles
    while True:
        try:
            c = nx.find_cycle(dag, orientation='original')
            e = random.choice(c)
            dag.remove_edge(e[0], e[1])
        except nx.NetworkXNoCycle:
            break
    return dag


def save_dag(path, adj, maps):
    np.save(path, adj)
    utils.save_dot(utils.adjmat2dot_map(adj, maps), path)


def save_adjusted_dag(path, adj, maps):
    dag = adjust_dag(utils.adjmat2dot_map(adj, maps))
    np.save(path, nx.adjacency_matrix(dag, [maps[key] for key in sorted(maps.keys())], dtype=int))
    utils.save_dot(dag, path)


def generate_config_dag_from_gnn(data_filename, epochs=None, th=None, backup=".bak"):
    config_path = os.path.join(CONFIG.path_dag_gnn, "DAG_from_GNN/config.py")

    for line in fileinput.FileInput(config_path, inplace=True, backup=backup):
        if "data_filename" in line:  # select which lines you care about
            print(line[:line.find("=") + 1] + " \"" + data_filename + "\"")
        elif epochs is not None and "epochs" in line:
            print(line[:line.find("=") + 1] + " " + str(epochs))
        elif th is not None and "graph_threshold" in line:
            print(line[:line.find("=") + 1] + " " + str(th))
        else:
            print(line, end='')

    return 0


def convert_adjcsv2dot(path_adj):
    df = pd.read_csv(path_adj)
    df.drop(df.columns[0], axis=1, inplace=True)
    mat = df.to_numpy()
    g = nx.from_numpy_array(mat, create_using=nx.DiGraph)

    for _, _, d in g.edges(data=True):  # remove weight
        d.clear()

    return nx.relabel_nodes(g, {n: df.columns[n] for n in range(len(mat))})


def dag_gnn_discovery(df_discovery, path_dag, th=0):
    path_data_disc = os.path.join(CONFIG.path_dag_gnn, "datasets", "mubench.csv")
    path_config = os.path.join(CONFIG.path_dag_gnn, "DAG_from_GNN", "config.py")
    path_results = os.path.join(CONFIG.path_dag_gnn, "results")
    path_adj_result = os.path.join(path_results, "final_adjacency_matrix.csv")

    if os.path.exists(path_results):
        shutil.rmtree(path_results)

    os.mkdir(path_results)
    df_discovery.to_csv(path_data_disc, index=False)

    generate_config_dag_from_gnn("mubench.csv", epochs=CONFIG.dag_gnn_epochs, th=th)

    subprocess.run(["cd " + CONFIG.path_dag_gnn + "; python3 -m DAG_from_GNN"], shell=True)

    dag = convert_adjcsv2dot(path_adj_result)
    dag_adjusted = adjust_dag(dag)
    utils.save_dot(dag_adjusted, path_dag)

    os.remove(path_config)
    os.rename(path_config + ".bak", path_config)
    os.remove(path_data_disc)
    shutil.rmtree(path_results)


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


def gen_configs_per_metric(df_discovery, loads_mapping, path_cgraph, path_configs, FAST=False, metrics=None):
    if metrics is None:
        metrics = CONFIG.all_metrics

    cg = nx.DiGraph(nx.nx_pydot.read_dot(path_cgraph))
    if cg.has_node("\\n"):  # TODO: BUG
        cg.remove_node("\\n")

    causal_model = gcm.StructuralCausalModel(cg)
    gcm.auto.assign_causal_mechanisms(causal_model, df_discovery, quality=gcm.auto.AssignmentQuality.GOOD)
    gcm.fit(causal_model, df_discovery)

    for ser in CONFIG.services:
        for met in metrics:
            conf_gen.generate_config(causal_model, df_discovery, ser,
                                     "{}_{}_{}.json".format(path_configs, met, ser),
                                     loads_mapping,
                                     metrics=[met],
                                     stability=0, nuser_limit=CONFIG.model_limit, FAST=FAST)


def create_min_anomalies_file(df, path, metrics=None):
    if metrics is None:
        metrics = CONFIG.all_metrics

    ns = list(set(df['NUSER']))

    out_j = {}
    for ser in CONFIG.services:
        out_j[ser] = {}
        ths = utils.calc_thresholds(df, ser)
        for met in metrics:
            for n in ns:
                icomb = []
                for load in CONFIG.loads:
                    for sr in CONFIG.SRs:
                        value = df[(df['NUSER'] == n) & (df['LOAD'] == load) & (df['SR'] == sr)][met + '_' + ser].mean()
                        if value > ths[met]:
                            icomb.append({'NUSER': n, 'LOAD': load, 'SR': sr})

                if len(icomb) > 0:
                    out_j[ser][met] = icomb
                    break

    with open(path, 'w') as f_out:
        json.dump(out_j, f_out)


def calc_metrics(df, path_configs, metrics=None):
    if metrics is None:
        metrics = CONFIG.all_metrics

    def calc_hamming_distance(conf_real, conf_pred):

        return abs(conf_real['NUSER'] - conf_pred['nusers'][0]) + (
            0 if conf_real['LOAD'] == conf_pred['loads'][0] else 1) + (
            0 if conf_real['SR'] == conf_pred['spawn_rates'][0] else 1)

    if not os.path.exists(CONFIG.path_minimal_configs_anomaly):
        create_min_anomalies_file(pd.read_csv(CONFIG.path_df), CONFIG.path_minimal_configs_anomaly)

    with open(CONFIG.path_minimal_configs_anomaly, 'r') as f_minimal_conf:
        minimal_confs = json.load(f_minimal_conf)

        true_positive = 0
        false_negative = 0
        false_positive = 0
        dists = []

        for ser in CONFIG.services:

            path_config = path_configs + "_" + ser + ".json"
            ths = utils.calc_thresholds(df, ser)

            if os.path.exists(path_config):
                with open(path_config, 'r') as f_c:
                    config = json.load(f_c)

                    df_config = df[(df['NUSER'] == config['nusers'][0]) &
                                   (df['LOAD'] == config['loads'][0]) &
                                   (df['SR'] == config['spawn_rates'][0])]

                    # X - Positive
                    for met in [m for m in config['anomalous_metrics'] if m in metrics]:
                        if df_config[met + "_" + ser].mean() > ths[met]:  # True - Positive
                            true_positive += 1
                            dists_c = np.zeros(len(minimal_confs[ser][met]), dtype=int)
                            for i, mc in enumerate(minimal_confs[ser][met]):
                                dists_c[i] = calc_hamming_distance(mc, config)
                            dists.append(min(dists_c))
                        else:  # False - Positive
                            false_positive += 1

            else:  # X - Negative
                false_negative = 0
                # check there is anomalies for service ser
                if all(met in minimal_confs[ser] for met in metrics):  # False - Negative
                    false_negative += 1

        precision = 0
        recall = 0

        if true_positive > 0:
            precision = true_positive / (true_positive + false_positive)

            if false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 1
        if len(dists) == 0:
            dists.append(-1)
        return {'precision': precision, 'recall': recall, 'mean_hamming_distance': np.mean(dists),
                'min_hamming_distance': int(np.min(dists)),
                'max_hamming_distance': int(np.max(dists))}


def merge_met_dict(met_dicts):
    out_dict = {}
    count_dict = {}

    for met in CONFIG.all_metrics:
        out_dict[met] = {
            "mean_precision": 0,
            "mean_recall": 0,
            "mean_mean_hamming_distance": 0,
            "mean_min_hamming_distance": 0,
            "mean_max_hamming_distance": 0
        }
        count_dict[met] = {
            "count_mean_hamming_distance": 0,
            "count_min_hamming_distance": 0,
            "count_max_hamming_distance": 0
        }
        for met_dict in met_dicts:
            out_dict[met]['mean_precision'] += met_dict[met]['precision'] / len(met_dicts)
            out_dict[met]['mean_recall'] += met_dict[met]['recall'] / len(met_dicts)

            if met_dict[met]['mean_hamming_distance'] >= 0:
                out_dict[met]['mean_mean_hamming_distance'] += met_dict[met]['mean_hamming_distance']
                count_dict[met]['count_mean_hamming_distance'] += 1
            if met_dict[met]['min_hamming_distance'] >= 0:
                out_dict[met]['mean_min_hamming_distance'] += met_dict[met]['min_hamming_distance']
                count_dict[met]['count_min_hamming_distance'] += 1
            if met_dict[met]['max_hamming_distance']:
                out_dict[met]['mean_max_hamming_distance'] += met_dict[met]['max_hamming_distance']
                count_dict[met]['count_max_hamming_distance'] += 1

        if count_dict[met]['count_mean_hamming_distance'] > 0:
            out_dict[met]['mean_mean_hamming_distance'] /= count_dict[met]['count_mean_hamming_distance']
        else:
            out_dict[met]['mean_mean_hamming_distance'] = -1

        if count_dict[met]['count_min_hamming_distance'] > 0:
            out_dict[met]['mean_min_hamming_distance'] /= count_dict[met]['count_min_hamming_distance']
        else:
            out_dict[met]['mean_min_hamming_distance'] = -1

        if count_dict[met]['count_max_hamming_distance'] > 0:
            out_dict[met]['mean_max_hamming_distance'] /= count_dict[met]['count_max_hamming_distance']
        else:
            out_dict[met]['mean_max_hamming_distance'] = -1
    return out_dict


def get_prior_mubench(columns, mets=None):
    return utils.get_generic_priorknorledge_mat(columns, CONFIG.services, {s: s for s in CONFIG.services},
                                                utils.get_architecture_from_wm(CONFIG.path_wm),
                                                num_load=len(CONFIG.loads), metrics=mets)
