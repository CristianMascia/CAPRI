import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def ___main__(path_df, path_png, services):
    df = pd.read_csv(path_df)
    nusers = list(set(df['NUSER']))
    nusers.sort()
    loads = list(set(df['LOAD']))
    loads.sort()
    SRs = list(set(df['SR']))
    SRs.sort()
    services.sort()
    df = df.groupby(['NUSER', 'LOAD', 'SR']).mean()

    text_fontsize = 14

    path_nuser_vs_met = os.path.join(path_png, "NUSER_vs_MET")

    if os.path.exists(path_png):
        shutil.rmtree(path_png)
    os.mkdir(path_png)
    os.mkdir(path_nuser_vs_met)

    for service in services:
        for met in ['RES_TIME', 'CPU', 'MEM']:
            fig, axs = plt.subplots(len(loads), len(SRs), figsize=(15, 10))
            fig.suptitle("{} for {}".format(met, service), fontsize=text_fontsize * 1.5)
            y_min = -1.
            y_max = -1.

            for l, load in enumerate(loads):
                for s, sr in enumerate(SRs):
                    y = [df.loc[(n, load, sr), met + "_" + service] for n in nusers]
                    axs[l, s].plot(nusers, y, marker='o')
                    plt.setp(axs[l, s], xticks=nusers)
                    if y_min == -1.:
                        y_min = min(y)
                    else:
                        y_min = min(y_min, min(y))
                    y_max = max(y_max, max(y))
            if y_min == y_max:
                continue

            for ax in axs.flat:
                offset = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - offset, y_max + offset)
                # ax.label_outer()

            for s, sr in enumerate(SRs):
                axs[0, s].set_title("SR=" + str(sr), fontsize=text_fontsize)

            for l, load in enumerate(loads):
                vertical_align_text(fig, axs[l, len(SRs) - 1], 1.1, "LOAD=" + load, text_fontsize, y_min, y_max)

            axs[int(len(loads) / 2), 0].set_ylabel(met, fontsize=text_fontsize)
            axs[len(loads) - 1, int(len(SRs) / 2)].set_xlabel('NUSER', fontsize=text_fontsize)
            fig.subplots_adjust(left=0.05, right=0.80)
            fig.savefig(os.path.join(path_nuser_vs_met, "{}_{}.png".format(service, met)))
            plt.close(fig)


def trainticket():
    with open("trainticket/mapping_service_request.json", 'r') as f_map:
        ___main__("trainticket/data/trainticket_df.csv", "trainticket/png", list(set(json.load(f_map).keys())))


def sockshop():
    with open("sockshop/mapping_service_request.json", 'r') as f_map:
        ___main__("sockshop/work/df.csv", "sockshop/work/png", list(set(json.load(f_map).keys())))


