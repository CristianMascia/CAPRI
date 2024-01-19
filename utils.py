import json
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import to_agraph


def threshold_matrix(mat, th=0.):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] > th:
                mat[i][j] = 1
            elif mat[i][j] < -th:
                mat[i][j] = -1
    return mat


def draw_prior_knwoledge_mat(mat, cols, path):
    G = nx.DiGraph()
    for c in range(len(cols)):
        if all(x == 0 for x in mat[c]):
            G.add_node(cols[c], color='red')
        else:
            G.add_node(cols[c], color='black')
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] == 1:
                G.add_edge(cols[j], cols[i])
    save_dot(G, path)


def adjmat2dot_map(adj_mat, map):
    dag = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    for _, _, d in dag.edges(data=True):  # rimuovo weight
        d.clear()
    nx.relabel_nodes(dag, map, copy=False)
    return dag


def adjmat2dot(adj_mat, cols):
    return adjmat2dot_map(adj_mat, {i: cols[i] for i in range(len(cols))})


def save_dot(G, path):
    A = to_agraph(G)
    A.write(path + ".dot")
    A.draw(path + ".png", prog='dot')


def hot_encode_array(a):
    out = {}
    for l in range(len(a)):
        out[a[l]] = [1 if j == l else 0 for j in range(len(a))]
    return out


def hot_encode_col(df, col):
    categories = list(dict.fromkeys(df[col]))
    col_one_hot = {}
    for l in range(len(categories)):
        col_one_hot[col + '_' + str(l)] = np.zeros(len(df.index), dtype=int)

    for r in range(len(df.index)):
        l = categories.index(df.loc[r, col])
        col_one_hot[col + '_' + str(l)][r] = 1

    df_one_hot = df.drop(columns=col)
    i = 1
    for k, v in col_one_hot.items():
        df_one_hot.insert(i, column=k, value=v)
        i += 1

    return df_one_hot


def hot_encode_col_mapping(df, col):
    categories = list(dict.fromkeys(df[col]))
    mapping = hot_encode_array(categories)

    col_one_hot = {}

    i = 0
    for k, v in mapping.items():
        col_one_hot[col + '_' + str(i)] = np.zeros(len(df.index), dtype=int)
        i += 1

    for r in range(len(df.index)):
        for j in range(len(mapping[df.loc[r, col]])):
            col_one_hot[col + '_' + str(j)][r] = mapping[df.loc[r, col]][j]

    df_one_hot = df.drop(columns=col)
    i = 1
    for k, v in col_one_hot.items():
        df_one_hot.insert(i, column=k, value=v)
        i += 1

    return df_one_hot, mapping


def get_generic_priorknorledge_mat(columns, services, mapping, architecture=None, num_load=3, metrics=None):
    if metrics is None:
        metrics = ['RES_TIME', 'CPU', 'MEM']

    maps = {i: columns[i] for i in range(0, len(columns))}
    inv_maps = {v: k for k, v in maps.items()}

    # 0: does not have a directed path to
    # 1: has a directed path to
    # -1 : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.

    mat = np.zeros([len(columns), len(columns)], dtype=int) - 1
    treatments = ['NUSER']
    if num_load == 1:
        treatments += ['LOAD']
    else:
        treatments += ['LOAD_' + str(i) for i in range(num_load)]
    treatments += ['SR']

    # impedisco archi entranti nei trattamenti
    for treat in treatments:
        for i in range(len(columns)):
            mat[inv_maps[treat]][i] = 0

    # le REQ/s dipendono da i trattamenti
    for ser in services:
        if "REQ/s_" + ser in columns:
            for treat in treatments:
                mat[inv_maps["REQ/s_" + ser]][inv_maps[treat]] = 1

    # le REQ/s incidono sulle metriche per ogni servizio
    for ser in services:
        if "REQ/s_" + ser in columns:
            for met in metrics:
                if met + "_" + ser in columns:
                    mat[inv_maps[met + "_" + ser]][inv_maps["REQ/s_" + ser]] = 1

    def connect_node(n1, n2, mets):
        for m in list(set(mets) & set(metrics)):
            mat[inv_maps["{}_{}".format(m, n2)]][inv_maps["{}_{}".format(m, n1)]] = 1

    if architecture is not None:
        for pod_start, pods_end in architecture.items():
            for pod_end in pods_end:
                services_by_pod_start = [s for s in mapping.keys() if mapping[s] == pod_start]
                services_by_pod_end = [s for s in mapping.keys() if mapping[s] == pod_end]
                if len(services_by_pod_start) == 0:  # pod_start non ha servizi esposti
                    if len(services_by_pod_end) == 0:  # pod_end non ha servizi esposti
                        connect_node(pod_start, pod_end, ['CPU', 'MEM'])
                    else:
                        for s in services_by_pod_end:
                            connect_node(pod_start, s, ['CPU', 'MEM'])
                else:
                    for ser_start in services_by_pod_start:
                        if len(services_by_pod_end) == 0:  # pod_end non ha servizi esposti
                            connect_node(ser_start, pod_end, ['CPU', 'MEM'])
                        else:
                            for ser_end in services_by_pod_end:
                                connect_node(ser_start, ser_end, ['RES_TIME', 'CPU', 'MEM'])
    return mat


def calc_threshold_met(df, service, met, filtered=False):
    d = df[df['NUSER'] == 1][met + "_" + service]
    if filtered:
        d = d[df["{}_{}".format(met, service)] > 0]
    return d.mean() + 3 * d.std()


# calcola solo le metriche che hanno colonna all'interno del dataset
def calc_thresholds(df, service, filtered=False):
    return {met: calc_threshold_met(df, service, met, filtered) for met in ['RES_TIME', 'CPU', 'MEM'] if
            "{}_{}".format(met, service) in df.columns}


def get_discovery_dataset(df):
    nusers = list(set(df['NUSER']))
    nusers_discovery = [nusers[i] for i in [0] + [i for i in range(2, len(nusers) - 1, 2)] + [len(nusers) - 1]]
    return hot_encode_col_mapping(df[df['NUSER'].isin(nusers_discovery)].reset_index(drop=True), 'LOAD')
