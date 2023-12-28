import os
import re
import numpy as np
import pandas as pd


def dockerstats_extractor(path, services):
    df = pd.read_csv(path, delim_whitespace=True)
    df = df[df["NAME"] != "NAME"].reset_index(drop=True)

    df['CPU(cores)'] = pd.to_numeric(df['CPU(cores)'].astype(str).str[:-1], errors='coerce')
    df['MEMORY(bytes)'] = pd.to_numeric(df['MEMORY(bytes)'].astype(str).str[:-2], errors='coerce')
    df = df.astype({'NAME': 'string'})

    # TODO: scrivere meglio, il replace non funziona
    for i in df.index:
        for ser in services:
            if ser in df.loc[i, 'NAME']:
                df.loc[i, 'NAME'] = ser
    df = df.drop(df[df['NAME'].isin((set(df['NAME']) - set(services)))].index)

    cpu = df.groupby(['NAME'], sort=False)['CPU(cores)'].mean()
    mem = df.groupby(['NAME'], sort=False)['MEMORY(bytes)'].mean()

    data = {'NAME': services, 'CPU AVG': np.zeros(len(services)), 'MEM AVG': np.zeros(len(services))}
    for i in range(len(services)):
        data['CPU AVG'][i] = cpu[data['NAME'][i]]
        data['MEM AVG'][i] = mem[data['NAME'][i]]

    return pd.DataFrame(data)


def locuststats_extractor(path):
    df = pd.read_csv(path).iloc[:-1, :]
    return df[['Name', 'Requests/s', 'Average Response Time']]


# mapping -> {pod : servizio, pod : servizio}
def merge_stats_df(doc_df, loc_df, mapping=None):
    data = {}
    for k, v in mapping.items():
        data['REQ/s_' + v] = loc_df[loc_df['Name'] == v]['Requests/s'].values[0]
        data['RES_TIME_' + v] = loc_df[loc_df['Name'] == k]['Average Response Time'].values[0]
        data['CPU_' + v] = doc_df[doc_df['NAME'] == k]['CPU AVG'].values[0]
        data['MEM_' + v] = doc_df[doc_df['NAME'] == k]['MEM AVG'].values[0]
    return data


def read_experiments(experiments_dir, mapping):
    services = list(mapping.values())
    cols = ['NUSER', 'LOAD', 'SR'] + [m + "_" + ser for ser in services for m in ['REQ/s', 'RES_TIME', 'CPU', 'MEM']]
    df_out = pd.DataFrame(columns=cols)

    edirs = os.listdir(experiments_dir)

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

                    df_dockerstats = dockerstats_extractor(os.path.join(path_dir, "mem_cpu_{}.txt".format(nrep)),
                                                           services)
                    df_locuststats = locuststats_extractor(os.path.join(path_dir, "esec_{}.csv_stats.csv".format(nrep)))
                    merged_row = merge_stats_df(df_dockerstats, df_locuststats, mapping)
                    merged_row['NUSER'] = nuser
                    merged_row['LOAD'] = ldir
                    merged_row['SR'] = sr

                    df_out = pd.concat([pd.DataFrame(merged_row, index=[0]), df_out.loc[:]]).reset_index(drop=True)

    df_out = df_out.sort_values(['NUSER', 'LOAD', 'SR'], inplace=False)[cols]
    return df_out
