import os
import shutil
import matplotlib.pyplot as plt
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


def ___main__(df, path_png, services, ths_filtered=False):
    nusers = list(set(df['NUSER']))
    nusers.sort()
    loads = list(set(df['LOAD']))
    loads.sort()
    SRs = list(set(df['SR']))
    SRs.sort()
    services.sort()
    df_grouped = df.groupby(['NUSER', 'LOAD', 'SR']).mean()

    text_fontsize = 14
    th_linewidth = 2
    unita_misura = {'CPU': 'cores', 'RES_TIME': 'ms', 'MEM': 'bytes'}
    label_met = {'CPU': 'CPU', 'RES_TIME': 'Response Time', 'MEM': 'Memory'}
    path_nuser_vs_met = os.path.join(path_png, "NUSER_vs_MET")

    if os.path.exists(path_png):
        shutil.rmtree(path_png)
    os.mkdir(path_png)
    os.mkdir(path_nuser_vs_met)

    for service in services:
        ths = utils.calc_thresholds(df, service, ths_filtered)
        for met in ['RES_TIME', 'CPU', 'MEM']:
            fig, axs = plt.subplots(len(loads), len(SRs), figsize=(23, 10))
            # fig.suptitle("{} for {}".format(met, service), fontsize=text_fontsize * 1.5)
            y_min = -1.
            y_max = -1.

            for l, load in enumerate(loads):
                for s, sr in enumerate(SRs):
                    y = [df_grouped.loc[(n, load, sr), met + "_" + service] for n in nusers]
                    axs[l, s].plot(nusers, y, marker='o')
                    plt.setp(axs[l, s], xticks=nusers)
                    axs[l, s].axhline(ths[met], color='red', linestyle='solid', linewidth=th_linewidth)
                    axs[l, s].text((max(nusers) - min(nusers)) * 0.3, ths[met], "TH={:.2f}".format(ths[met]))

                    # trans = transforms.blended_transform_factory(
                    #    axs[l, s].get_yticklabels()[0].get_transform(), axs[l, s].transData)
                    # axs[l, s].text(0, ths[met], "{:.2f}".format(ths[met]), color="red", transform=trans,
                    #               ha="right", va="center")

                    if y_min == -1.:
                        y_min = min(y)
                    else:
                        y_min = min(y_min, min(y))
                    y_max = max(y_max, max(y), ths[met])

            for ax in axs.flat:
                offset = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - offset, y_max + offset)
                ax.yaxis.labelpad = 10
                ax.xaxis.labelpad = 10
                # ax.label_outer()

            for s, sr in enumerate(SRs):
                axs[0, s].set_title("Spawn Rate=" + str(sr), fontsize=text_fontsize)

            for l, load in enumerate(loads):
                vertical_align_text(fig, axs[l, len(SRs) - 1], 1.1, "LOAD=" + load.replace("_", " "), text_fontsize,
                                    y_min, y_max)

            axs[int(len(loads) / 2), 0].set_ylabel("{}[{}]".format(label_met[met], unita_misura[met]),
                                                   fontsize=text_fontsize)
            axs[len(loads) - 1, int(len(SRs) / 2)].set_xlabel('Users Size', fontsize=text_fontsize)
            fig.subplots_adjust(left=0.03, right=0.85)
            fig.savefig(os.path.join(path_nuser_vs_met, "{}_{}.png".format(service, met)))
            plt.close(fig)
