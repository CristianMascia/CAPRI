import os
import re
import numpy as np
import pandas as pd


def rename_startwith(p, pods):
    pods = sorted(pods, key=len, reverse=True)
    for pod in pods:
        if p.startswith(pod):
            return pod
    return p


def dockerstats_extractor(path, pods, pod_renaming_func=rename_startwith):
    df = pd.read_csv(path, sep='\\s+')

    df = df[df["NAME"] != "NAME"].reset_index(drop=True)

    df['CPU(cores)'] = pd.to_numeric(df['CPU(cores)'].astype(str).str[:-1], errors='coerce')
    df['MEMORY(bytes)'] = pd.to_numeric(df['MEMORY(bytes)'].astype(str).str[:-2], errors='coerce')
    df = df.astype({'NAME': 'string'})

    cpu = df.groupby(['NAME'], sort=False)['CPU(cores)'].mean()
    mem = df.groupby(['NAME'], sort=False)['MEMORY(bytes)'].mean()

    if pod_renaming_func is not None:
        cpu = cpu.set_axis([pod_renaming_func(i, pods) for i in cpu.index], copy=False)
        mem = mem.set_axis([pod_renaming_func(i, pods) for i in mem.index], copy=False)

    data = {'NAME': pods, 'CPU AVG': np.zeros(len(pods)), 'MEM AVG': np.zeros(len(pods))}

    for i in range(len(pods)):
        data['CPU AVG'][i] = cpu[data['NAME'][i]].mean()  # BUG
        data['MEM AVG'][i] = mem[data['NAME'][i]].mean()

    return pd.DataFrame(data)


def locuststats_extractor(path):
    df = pd.read_csv(path).iloc[:-1, :]
    return df[['Name', 'Requests/s', 'Average Response Time']]


# mapping -> {servizio : pod}
def merge_stats_df(doc_df, loc_df, mapping, pods_):
    data = {}
    services = list(set(mapping.keys()))
    pods = pods_.copy()

    for ser in services:
        if len(loc_df[loc_df['Name'] == ser]) > 0:
            data['REQ/s_' + ser] = loc_df[loc_df['Name'] == ser]['Requests/s'].values[0]
            data['RES_TIME_' + ser] = loc_df[loc_df['Name'] == ser]['Average Response Time'].values[0]
        else:  # MISSING VALUE -> richiesta mai eseguita in quell'esperimento
            data['REQ/s_' + ser] = 0
            data['RES_TIME_' + ser] = 0

        data['CPU_' + ser] = doc_df[doc_df['NAME'] == mapping[ser]]['CPU AVG'].values[0]
        data['MEM_' + ser] = doc_df[doc_df['NAME'] == mapping[ser]]['MEM AVG'].values[0]

        if mapping[ser] in pods:
            pods.remove(mapping[ser])

    for pod in pods:
        data['CPU_' + pod] = doc_df[doc_df['NAME'] == pod]['CPU AVG'].values[0]
        data['MEM_' + pod] = doc_df[doc_df['NAME'] == pod]['MEM AVG'].values[0]
    return data


def read_experiments(experiments_dir, mapping, pods, pod_renaming_func=rename_startwith):
    services = list(mapping.keys())
    cols = ['NUSER', 'LOAD', 'SR'] + [m + "_" + ser for ser in services for m in ['REQ/s', 'RES_TIME', 'CPU', 'MEM']]
    cols += [(m + "_" + pod) for pod in pods for m in ['CPU', 'MEM'] if pod not in mapping.values()]

    df_out = pd.DataFrame(columns=cols)

    edirs = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]

    for edir in edirs:
        sr = int(edir[edir.rindex('_') + 1:])
        udirs = os.listdir(os.path.join(experiments_dir, edir))

        for udir in udirs:
            nuser = int(udir[udir.rindex('_') + 1:])
            ldirs = os.listdir(os.path.join(experiments_dir, edir, udir))
            for ldir in ldirs:
                path_dir = os.path.join(experiments_dir, edir, udir, ldir)
                dockerstats = [f for f in os.listdir(path_dir) if re.match(r'mem_cpu_[1-9].txt', f)]

                for dstat in dockerstats:
                    nrep = int(dstat[dstat.rindex('_') + 1:dstat.rindex('.')])
                    df_dockerstats = dockerstats_extractor(os.path.join(path_dir, "mem_cpu_{}.txt".format(nrep)), pods,
                                                           pod_renaming_func)

                    df_locuststats = locuststats_extractor(os.path.join(path_dir, "esec_{}.csv_stats.csv".format(nrep)))
                    merged_row = merge_stats_df(df_dockerstats, df_locuststats, mapping, pods)
                    merged_row['NUSER'] = nuser
                    merged_row['LOAD'] = ldir
                    merged_row['SR'] = sr
                    df_out = pd.concat([pd.DataFrame(merged_row, index=[0]), df_out.loc[:]]).reset_index(drop=True)

    df_out = df_out.sort_values(['NUSER', 'LOAD', 'SR'], inplace=False)[cols]
    return df_out.reset_index(drop=True)
