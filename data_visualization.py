import json
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import data_preparation
import utils
import matplotlib.transforms as transforms


def vertical_align_text(fig, ax, x, text, fontsize, y_min, y_max):
    tobj = ax.text(x, 0.5, text, fontsize=fontsize, transform=ax.transAxes)
    r = fig.canvas.get_renderer()
    transf = ax.transData.inverted()
    bb = tobj.get_window_extent(renderer=r)
    bb_datacoords = bb.transformed(transf)
    tobj.remove()

    if y_max > y_min:
        offset = ((bb_datacoords.y1 - bb_datacoords.y0) / (y_max - y_min)) / 4
    else:
        offset = 0
    ax.text(x, 0.5 - offset, text, fontsize=fontsize, transform=ax.transAxes)


def ___main__(df, path_png, services):
    nusers = list(set(df['NUSER']))
    nusers.sort()
    loads = list(set(df['LOAD']))
    loads.sort()
    SRs = list(set(df['SR']))
    SRs.sort()
    services.sort()
    df_grouped = df.groupby(['NUSER', 'LOAD', 'SR']).mean()

    text_fontsize = 14

    path_nuser_vs_met = os.path.join(path_png, "NUSER_vs_MET")

    if os.path.exists(path_png):
        shutil.rmtree(path_png)
    os.mkdir(path_png)
    os.mkdir(path_nuser_vs_met)

    for service in services:
        ths = utils.calc_thresholds(df, service)
        for met in ['RES_TIME', 'CPU', 'MEM']:
            fig, axs = plt.subplots(len(loads), len(SRs), figsize=(23, 10))
            fig.suptitle("{} for {}".format(met, service), fontsize=text_fontsize * 1.5)
            y_min = -1.
            y_max = -1.

            for l, load in enumerate(loads):
                for s, sr in enumerate(SRs):
                    y = [df_grouped.loc[(n, load, sr), met + "_" + service] for n in nusers]
                    axs[l, s].plot(nusers, y, marker='o')
                    plt.setp(axs[l, s], xticks=nusers)
                    axs[l, s].axhline(ths[met], color='red', linestyle='solid')

                    trans = transforms.blended_transform_factory(
                        axs[l, s].get_yticklabels()[0].get_transform(), axs[l, s].transData)
                    axs[l, s].text(0, ths[met], "{:.0f}".format(ths[met]), color="red", transform=trans,
                                   ha="right", va="center")

                    if y_min == -1.:
                        y_min = min(y)
                    else:
                        y_min = min(y_min, min(y))
                    y_max = max(y_max, max(y), ths[met])
            if y_min == y_max:
                continue

            for ax in axs.flat:
                offset = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - offset, y_max + offset)
                # ax.label_outer()

            for s, sr in enumerate(SRs):
                axs[0, s].set_title("SR=" + str(sr), fontsize=text_fontsize)

            for l, load in enumerate(loads):
                vertical_align_text(fig, axs[l, len(SRs) - 1], 1.1, "LOAD=" + load.replace("-", " "), text_fontsize,
                                    y_min, y_max)

            axs[int(len(loads) / 2), 0].set_ylabel(met, fontsize=text_fontsize)
            axs[len(loads) - 1, int(len(SRs) / 2)].set_xlabel('NUSER', fontsize=text_fontsize)
            fig.subplots_adjust(left=0.03, right=0.85)
            fig.savefig(os.path.join(path_nuser_vs_met, "{}_{}.png".format(service, met)))
            plt.close(fig)


def mubench():
    path_system = "mubench"
    path_df = os.path.join(path_system, "data", "mubench_df.csv")
    path_png = os.path.join(path_system, "png")

    df = pd.read_csv(path_df)
    df_disc = df[df['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
    ___main__(df_disc, path_png, ['s' + str(i) for i in range(10)])


def trainticket():
    path_system = "trainticket"
    path_mapping = os.path.join(path_system, "mapping_service_request.json")
    path_data = os.path.join(path_system, "data")
    path_df = os.path.join(path_data, "trainticket_df.csv")
    path_png = os.path.join(path_system, "png")
    path_pods = os.path.join(path_system, "pods.txt")

    with open(path_mapping, 'r') as f_map:
        mapping = json.load(f_map)

        if not os.path.exists(path_df):
            with open(path_pods, 'r') as f_pods:
                pods = [p.replace("\n", "") for p in f_pods.readlines()]
                data_preparation.read_experiments(path_data, mapping, pods,
                                                  data_preparation.rename_startwith).to_csv(path_df, index=False)
        df = pd.read_csv(path_df)
        ___main__(df, path_png, list(set(mapping.keys())))


def sockshop():
    path_system = "sockshop"
    path_mapping = os.path.join(path_system, "mapping_service_request.json")
    path_data = os.path.join(path_system, "data")
    path_df = os.path.join(path_data, "sockshop_df.csv")
    path_png = os.path.join(path_system, "png")
    path_pods = os.path.join(path_system, "pods.txt")

    with open(path_mapping, 'r') as f_map:
        mapping = json.load(f_map)

        if not os.path.exists(path_df):
            with open(path_pods, 'r') as f_pods:
                pods = [p.replace("\n", "") for p in f_pods.readlines()]
                data_preparation.read_experiments(path_data, mapping, pods,
                                                  data_preparation.rename_startwith).to_csv(path_df, index=False)
        df = pd.read_csv(path_df)
        ___main__(df, path_png, list(set(mapping.keys())))
