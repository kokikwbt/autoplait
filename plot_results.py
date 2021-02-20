import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn import preprocessing


sns.set_style('whitegrid')
sns.set_palette('muted')


def plot_segments(data, regimes, fp=None, fs=18):

    cmap = cm.Set1
    data = preprocessing.minmax_scale(data)

    fig, ax = plt.subplots(2, figsize=(10, 8))

    # Original data

    ax[0].plot(data)
    ax[0].set_title('Original data', fontsize=fs+2)
    ax[0].set_xlabel('Time', fontsize=fs)
    ax[0].set_ylabel('Value', fontsize=fs)
    ax[0].tick_params(axis='both', labelsize=fs-2)

    # Segmentation result

    for i, R in enumerate(regimes):
        for st, dt in R.S:
            x = [st, st + dt - 1]
            y = [i, i]
            ax[1].plot(x, y, color=cmap(i), linewidth=10)

    ax[1].set_title('Segmentation', fontsize=fs+2)
    ax[1].set_ylabel('Regime ID', fontsize=fs)
    ax[1].tick_params(axis='both', labelsize=fs-2)

    fig.tight_layout()

    if fp is not None:
        fig.savefig(fp + 'result.pdf')
