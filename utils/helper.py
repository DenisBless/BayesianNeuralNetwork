import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class HistGif:
    def __init__(self):
        self.writer = PillowWriter(fps=25)
        save_path = "test.gif"

    @staticmethod
    def update_hist(num, data):
        plt.cla()
        plt.xlim(-0.2, 0.2)
        plt.xticks(np.linspace(-0.2, 0.2, 10))
        plt.hist(x=data[:, num], bins=20, color='dodgerblue', alpha=0.7, rwidth=0.85)

    def make_gif(self, data, nframes):
        fig = plt.figure()
        plt.xlim(-0.2, 0.2)
        plt.xticks(np.linspace(-0.2, 0.2, 10))
        hist = plt.hist(x=data[:, 0], bins=20, color='dodgerblue', alpha=0.7, rwidth=0.85)
        animation = FuncAnimation(fig, HistGif.update_hist, nframes, fargs=(data,))
        animation.save("test.gif", dpi=100, writer=self.writer)


class PltGif:
    def __init__(self):
        self.writer = PillowWriter(fps=25)
        save_path = "test.gif"

    @staticmethod
    def update_hist(num, data):
        plt.cla()
        plt.xlim(-0.2, 0.2)
        plt.xticks(np.linspace(-0.2, 0.2, 10))
        plt.plot(x.detach().numpy(), results.mean(dim=0).detach().numpy(),
         linewidth=3, c='dodgerblue')

    def make_gif(self, data, nframes):
        fig = plt.figure()
        plt.xlim(-0.2, 0.2)
        plt.xticks(np.linspace(-0.2, 0.2, 10))
        hist = plt.hist(x=data[:, 0], bins=20, color='dodgerblue', alpha=0.7, rwidth=0.85)
        animation = FuncAnimation(fig, HistGif.update_hist, nframes, fargs=(data,))
        animation.save("test.gif", dpi=100, writer=self.writer)


def eval_net(net, x, nsamples):
    results = torch.zeros([x.shape[0], nsamples])
    for i in range(nsamples):
        X_eval = x.reshape(-1, 1)
        Y_pred = net(X_eval)
        results[:, i] = Y_pred.reshape(-1)
    mean = results.mean(dim=0).detach().numpy()
    std = results.std(dim=0).detach().numpy()
    return mean, std
