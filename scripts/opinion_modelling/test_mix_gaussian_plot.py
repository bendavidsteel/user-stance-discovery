import os

import matplotlib.pyplot as plt
import numpy as np

import landscape

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..', '..')

    probs = landscape.sum_gaussians(np.array([-0.9, 0.9, 0.1]), np.array([0.01, 0.01, 0.0]), 25)
    probs_2d = landscape.sum_multi_gaussians(np.array([[-0.9, 0.9], [0.9, -0.9], [0.5, 0.5]]), np.array([[0.05, 0.01], [0.01, 0.05], [0., 0.]]), 25)

    fig, axes = plt.subplots(ncols=2, nrows=1)

    axes[0].bar(np.linspace(-1, 1, 25), probs, width=0.08)
    axes[1].imshow(probs_2d, cmap='hot', interpolation='nearest', extent=[-1, 1, -1, 1], origin='lower')

    fig.savefig(os.path.join(root_dir_path, 'figs', 'test_mix_gaussian_plot.png'))

if __name__ == '__main__':
    main()