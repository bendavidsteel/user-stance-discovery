import os

import numpy as np
import matplotlib.pyplot as plt



def main():
    xs = np.linspace(-3, 3, 1000)
    ys = np.linspace(-3, 3, 1000)
    s = np.zeros((len(xs), len(ys), 2))
    s[:, :, 0] = np.tile(xs, (len(ys), 1))
    s[:, :, 1] = np.tile(ys, (len(xs), 1)).T

    a = np.array([0, -1])
    b = np.array([0, 1])
    a_b_dist = np.linalg.norm(a - b)

    a_dist = np.linalg.norm(s - a, axis=2)
    b_dist = np.linalg.norm(s - b, axis=2)
    m_along = (a_dist / (a_dist + b_dist))
    m_away = np.minimum(-0.5, -1.5 * (1 * (np.e ** (((a_dist + b_dist) / a_b_dist) - 1) - 1) / ((np.e ** (((a_dist + b_dist) / a_b_dist) - 1) - 1) + 1)))
    m = m_along + m_away
    plt.matshow(m)
    plt.colorbar()

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    fig_path = os.path.join(root_dir_path, 'figs', 'movement.png')
    plt.savefig(fig_path)

if __name__ == '__main__':
    main()