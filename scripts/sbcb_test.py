import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def main():
    n = 10
    p = np.arange(-1, 1, 0.1)
    d = np.arange(0.0, 10.0, 0.1)

    P, D = np.meshgrid(p, d)
    # nu = (n ** np.abs(P) * (1 / n) ** (1 - np.abs(P))) + (P * (np.e ** (-D*np.abs(P))))
    nu = D ** -P / (D ** -P + 0.3 ** -P)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(P, D, nu)
    plt.xlabel('p')
    plt.ylabel('d')

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    fig_path = os.path.join(root_dir_path, 'figs', 'sbcd_test.png')
    fig.savefig(fig_path)

if __name__ == '__main__':
    main()