import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

import opinion_datasets

def get_gradient(x):
    x_grad = np.gradient(x)
    gradient = np.empty(x.shape + (x.ndim,), dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        gradient[:, :, k] = grad_k
    return gradient

def get_hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty(x.shape + (x.ndim, x.ndim), dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[:, :, k, l] = grad_kl
    return hessian

def get_maxima(x):
    grad_x = np.gradient(x, axis=0)
    grad_y = np.gradient(x, axis=1)
    sgn_grad_x = np.sign(grad_x)
    sgn_grad_y = np.sign(grad_y)
    grad_sgn_grad_x = np.gradient(sgn_grad_x, axis=0)
    grad_sgn_grad_y = np.gradient(sgn_grad_y, axis=1)
    maxima = np.logical_and(grad_sgn_grad_x == -2, grad_sgn_grad_y == -2)
    return maxima

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    num_people = 10
    max_time_step = 10
    dataset = opinion_datasets.GenerativeOpinionTimelineDataset(num_people=100, max_time_step=max_time_step)

    all_opinions = []
    for i in range(max_time_step):
        opinions, users = dataset[i]
        all_opinions.extend(opinions)

    all_opinions = np.array(all_opinions)

    grid_size = 100

    # create points for each point in the mesh of the multi dimensional opinion space
    num_opinions = all_opinions.shape[1]
    opinion_points = np.meshgrid(*[np.linspace(-1, 1, grid_size) for i in range(num_opinions)])
    opinion_points = np.array([opinion_point.ravel() for opinion_point in opinion_points]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(all_opinions)
    log_dens = kde.score_samples(opinion_points)
    dens = np.exp(log_dens)

    density_grid = dens.reshape((grid_size,)*num_opinions)

    # pad grid in order to find maxima at the edges
    # density_grid = np.pad(density_grid, 1, mode='constant', constant_values=0)

    gradient = get_gradient(density_grid)
    hessian = get_hessian(density_grid)
    eigenvalues = np.linalg.eigvals(hessian)
    maxima = get_maxima(density_grid)

    # unpad to get the original grid
    # density_grid = density_grid[1:-1, 1:-1]
    # gradient = gradient[1:-1, 1:-1]
    # hessian = hessian[1:-1, 1:-1]
    # eigenvalues = eigenvalues[1:-1, 1:-1]

    # get points where the gradient is zero and the eigenvalues of the hessian are negative
    inflection_indices = np.argwhere(np.all(np.isclose(gradient, 0), axis=-1))
    neg_eigen_indices = np.argwhere(np.all(eigenvalues < 0, axis=-1))
    maxima_indices = np.intersect1d(inflection_indices, neg_eigen_indices)

    # plot maxima strength histogram
    maxima_amp = dens[maxima_indices]
    fig, axes = plt.subplots(nrows=3)

    axes[0].imshow(density_grid, extent=[-1,1,-1,1], origin='lower', cmap='viridis')
    axes[0].scatter(all_opinions[:,0], all_opinions[:,1], s=10, c='r', label="Opinions")
    axes[0].scatter(opinion_points[maxima_indices,0], opinion_points[maxima_indices,1], s=10, c='b', label="Maxima")
    axes[0].set_title("Opinion Landscape")
    axes[0].legend()

    axes[1].imshow(maxima, extent=[-1,1,-1,1], origin='lower', cmap='viridis')

    axes[2].hist(maxima_amp, bins=100)
    axes[2].set_title("Maxima Strength Histogram")
    axes[2].set_xlabel("Maxima Strength")
    axes[2].set_ylabel("Frequency")

    fig.savefig(os.path.join(root_dir_path, "figs", "opinion_landscape.png"))

    # get only maxima with a magnitude greater than 0.1
    maxima_indices = maxima_indices[maxima_amp > 0.1]

    user_attractors = {}
    for opinion_snapshots in all_opinions:
        # find nearest attractor
        for user_id, opinion in enumerate(opinion_snapshots):
            nearest_attractor = None
            user_attractors[user_id] = nearest_attractor

if __name__ == '__main__':
    main()