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
import performance_evaluator
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


def calc_metrics(df, path_configs_dir, model_name, metrics=None):
    if metrics is None:
        metrics = CONFIG.all_metrics

    def get_config(ser, met):
        path_conf = os.path.join(path_configs_dir, "{}_{}_{}.json".format(model_name, met, ser))
        if os.path.isfile(path_conf):
            with open(path_conf, 'r') as f_c:
                config = json.load(f_c)
                df_config = df[(df['NUSER'] == config['nusers'][0]) &
                               (df['LOAD'] == config['loads'][0]) &
                               (df['SR'] == config['spawn_rates'][0])]
                return config, df_config[met + "_" + ser].mean()
        else:
            return None, None

    return performance_evaluator.calc_metrics(df, get_config, CONFIG.services, CONFIG.model_limit, metrics=metrics)


def merge_met_dict(met_dicts):
    out_dict = {}
    count_dict = {}

    for met in CONFIG.all_metrics:
        out_dict[met] = {
            "mean_precision": 0,
            "mean_recall": 0,
            "mean_mhd_pos": 0,
            "mean_mhd_false": 0
        }
        count_dict[met] = {
            "count_mhd_pos": 0,
            "count_mhd_false": 0
        }
        for met_dict in met_dicts:
            out_dict[met]['mean_precision'] += met_dict[met]['precision'] / len(met_dicts)
            out_dict[met]['mean_recall'] += met_dict[met]['recall'] / len(met_dicts)

            for t in ['pos', 'false']:
                if met_dict[met]['mhd_' + t] >= 0:
                    out_dict[met]['mean_mhd_' + t] += met_dict[met]['mhd_' + t]
                    count_dict[met]['count_mhd_' + t] += 1

        for t in ['pos', 'false']:
            if count_dict[met]['count_mhd_' + t] > 0:
                out_dict[met]['mean_mhd_' + t] /= count_dict[met]['count_mhd_' + t]
            else:
                out_dict[met]['mean_mhd_' + t] = -1
    return out_dict


def get_prior_mubench(columns, mets=None):
    return utils.get_generic_priorknorledge_mat(columns, CONFIG.services, {s: s for s in CONFIG.services},
                                                utils.get_architecture_from_wm(CONFIG.path_wm),
                                                num_load=len(CONFIG.loads), metrics=mets)
