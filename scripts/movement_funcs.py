import os

import numpy as np
import matplotlib.pyplot as plt



def main():
    xs = np.linspace(-3, 3, 1000)
    ys = np.linspace(3, -3, 1000)
    s = np.zeros((len(xs), len(ys), 2))
    s[:, :, 0] = np.tile(xs, (len(ys), 1))
    s[:, :, 1] = np.tile(ys, (len(xs), 1)).T

    prev = np.array([0, -1])
    inter = np.array([0, 1])
    prev_inter_dist = np.linalg.norm(prev - inter)

    prev_dist = np.linalg.norm(s - prev, axis=2)
    inter_dist = np.linalg.norm(s - inter, axis=2)
    #m_along = (a_b_dist - b_dist) / a_b_dist
    #m_along = np.clip((a_dist / a_b_dist) * (a_dist**2 + a_b_dist**2 - b_dist**2) / (2 * a_dist * a_b_dist), 0, 1)
    m_along = 1 / (1 + (np.e ** (inter_dist - prev_dist)))
    #m_away = 1 / np.e ** (((a_dist + b_dist) / a_b_dist) - 1)
    from_line = ((prev_dist + inter_dist) / prev_inter_dist) - 1
    m_away = np.e ** from_line / (np.e ** from_line + 1)
    #m[np.abs(m) < 0.01] = -10

    fig, axes = plt.subplots(nrows=1, ncols=2)

    im1 = axes[0].imshow(m_along, extent=(-3, 3, -3, 3))
    axes[0].plot(0, -1, 'ro', label='Prev')
    axes[0].plot(0, 1, 'go', label='Interacted')
    axes[0].set_title('Between')
    axes[0].legend()

    im2 = axes[1].imshow(m_away, extent=(-3, 3, -3, 3))
    axes[1].plot(0, -1, 'ro', label='Prev')
    axes[1].plot(0, 1, 'go', label='Interacted')
    axes[1].set_title('Away')

    plt.tight_layout()

    fig.colorbar(im1, ax=axes.ravel().tolist())

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    fig_path = os.path.join(root_dir_path, 'figs', 'movement.png')
    fig.savefig(fig_path)

if __name__ == '__main__':
    main()