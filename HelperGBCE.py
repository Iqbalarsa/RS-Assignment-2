import numpy as np
import matplotlib.pyplot as plt


class LearningCurvePlot:
    def __init__(self, title=None, xlabel="Epoch", ylabel="Metric"):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, x, y, label=None):
        if label is not None:
            self.ax.plot(x, y, label=label)
        else:
            self.ax.plot(x, y)

    def add_curve_with_error(self, x, y, yerr, label=None, alpha=0.2):
        self.ax.plot(x, y, label=label)
        self.ax.fill_between(x, y - yerr, y + yerr, alpha=alpha)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label=None):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='plot.png'):
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.savefig(name, dpi=300, bbox_inches='tight')
        plt.close(self.fig)


def smooth(y, window=5):
    y = np.asarray(y, dtype=float)
    if len(y) < window or window <= 1:
        return y

    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")