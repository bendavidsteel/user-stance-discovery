import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

import opinion_datasets, estimate



def get_gradient(x):
    x_grad = np.gradient(x)
    gradient = np.empty(x.shape + (x.ndim,), dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        gradient[tuple([np.s_[:] for _ in range(x.ndim)] + [np.s_[k]])] = grad_k
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
            hessian[tuple([np.s_[:] for _ in range(x.ndim)] + [np.s_[k], np.s_[l]])] = grad_kl
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

def sum_gaussians(opinion_means, opinion_variances, i, num_bins):
    # Create a linspace array which will be reused
    linspace_array = np.linspace(-1, 1, num_bins)

    # Identify valid indices where none of the values are NaN
    valid_indices = ~np.isnan(opinion_means[:, i]) & ~np.isnan(opinion_variances[:, i]) & (opinion_variances[:, i] > 0)

    # Extract the valid means and variances
    valid_means = opinion_means[valid_indices, i]
    valid_variances = opinion_variances[valid_indices, i]

    # Calculate the probability densities for the valid pairs
    first_term = 1 / (valid_variances * np.sqrt(2 * np.pi)) * np.exp(-(linspace_array[:, None] - valid_means) ** 2 / (2 * valid_variances))
    prob_density = np.sum(first_term, axis=1)

    # add point masses
    valid_indices = ~np.isnan(opinion_means[:, i]) & ~np.isnan(opinion_variances[:, i]) & (opinion_variances[:, i] == 0)
    valid_means = opinion_means[valid_indices, i]
    for mean in valid_means:
        prob_density[np.argmin(np.abs(linspace_array - mean))] += 1

    prob_density /= np.sum(prob_density)
    return prob_density

def sum_multi_gaussians(opinion_means, opinion_variances, i, j, num_bins):
    # Create a linspace array which will be reused
    linspace_array = np.linspace(-1, 1, num_bins)

    # Identify valid indices where none of the values are NaN
    valid_indices = ~np.isnan(opinion_means[:, j]) & ~np.isnan(opinion_variances[:, j]) & ~np.isnan(opinion_means[:, i]) & ~np.isnan(opinion_variances[:, i]) & (opinion_variances[:, j] > 0) & (opinion_variances[:, i] > 0)

    # Extract the valid means and variances
    valid_means1 = opinion_means[valid_indices, i]
    valid_variances1 = opinion_variances[valid_indices, i]
    valid_means2 = opinion_means[valid_indices, j]
    valid_variances2 = opinion_variances[valid_indices, j]

    # Calculate the probability densities for the valid pairs
    first_term = 1 / (valid_variances1 * np.sqrt(2 * np.pi)) * np.exp(-(linspace_array[:, None] - valid_means1) ** 2 / (2 * valid_variances1))
    second_terms = 1 / (valid_variances2 * np.sqrt(2 * np.pi)) * np.exp(-(linspace_array[:, None] - valid_means2) ** 2 / (2 * valid_variances2))

    # Sum over the second dimension to get the total probability density
    first_term = np.sum(first_term, axis=1)
    second_terms = np.sum(second_terms, axis=1)
    prob_density = first_term[:, None] * second_terms[None, :]

    # add point masses
    valid_indices = ~np.isnan(opinion_means[:, j]) & ~np.isnan(opinion_variances[:, j]) & ~np.isnan(opinion_means[:, i]) & ~np.isnan(opinion_variances[:, i]) & (opinion_variances[:, j] == 0) & (opinion_variances[:, i] == 0)
    valid_means1 = opinion_means[valid_indices, i]
    valid_means2 = opinion_means[valid_indices, j]
    for mean1, mean2 in zip(valid_means1, valid_means2):
        prob_density[np.argmin(np.abs(linspace_array - mean1)), np.argmin(np.abs(linspace_array - mean2))] += 1

    prob_density /= np.sum(prob_density)
    return prob_density

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    dataset_name = "reddit"
    aggregation = "weighted_mean"

    if dataset_name == "reddit":
        dataset = opinion_datasets.RedditOpinionTimelineDataset(aggregation=aggregation)
    elif dataset_name == "generative":
        num_people = 10
        max_time_step = 10
        num_opinions = 3
        dataset = opinion_datasets.GenerativeOpinionTimelineDataset(num_people=num_people, max_time_step=max_time_step, num_opinions=num_opinions)

    num_people = dataset.num_people
    max_time_step = dataset.max_time_step

    all_opinions = []
    all_users = []
    opinion_means, opinion_variances, users = dataset[max_time_step] 

    if False:
        grid_size = 10

        # create points for each point in the mesh of the multi dimensional opinion space
        num_opinions = all_opinions.shape[1]
        opinion_points = np.meshgrid(*[np.linspace(-1, 1, grid_size, dtype=np.float16) for i in range(num_opinions)])
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

    
    num_bins = 25
    # plot pairplot from opinion means and variances
    user_opinion_df = pd.DataFrame(opinion_means, columns=dataset.stance_columns)
    user_opinion_df["user"] = users
    g = sns.pairplot(user_opinion_df, diag_kind = "hist", diag_kws = {'bins':num_bins})
    g.savefig(os.path.join(root_dir_path, "figs", "opinion_pairplot.png"))

    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), ncols=len(dataset.stance_columns), figsize=(20, 20))
    for i, stance1 in enumerate(dataset.stance_columns):
        for j, stance2 in enumerate(dataset.stance_columns):
            if i == j:
                prob_density = sum_gaussians(opinion_means, opinion_variances, i, num_bins)
                axes[i,j].plot(np.linspace(-1, 1, num_bins), prob_density)
                axes[i,j].set_xlabel(stance1)
                axes[i,j].set_ylabel("Probability Density")
            else:
                prob_density = sum_multi_gaussians(opinion_means, opinion_variances, i, j, num_bins)
                axes[i,j].imshow(prob_density, extent=[-1,1,-1,1], origin='lower', cmap='viridis')
                axes[i,j].set_xlabel(stance1)
                axes[i,j].set_ylabel(stance2)
    fig.savefig(os.path.join(root_dir_path, "figs", "opinion_distribution_pairplot.png"))


    get_examples = True
    if get_examples:
        # get examples of moderate and extreme users for each opinion
        for stance in dataset.stance_columns:
            extreme_favor_user = user_opinion_df.sort_values(by=stance, ascending=False).head(1).iloc[0]
            extreme_against_user = user_opinion_df.sort_values(by=stance, ascending=True).head(1).iloc[0]
            # get user with stance closest to 0
            moderate_user = user_opinion_df.iloc[user_opinion_df[stance].abs().argsort()[0]]

            def print_comments(comment_df, stance):
                comment_df = comment_df[comment_df[stance].notna()]
                comment_df = comment_df.sample(3) if len(comment_df) > 3 else comment_df
                for idx, row in comment_df.iterrows():
                    d = {
                        "stance": row[stance],
                        "comment": row["comment"]
                    }
                    print(json.dumps(d, indent=4))

            print(f"Stance: {stance}")
            extreme_favor_comment_df = dataset.get_user_comments(extreme_favor_user['user'])
            print(f"Extreme Favor User: {extreme_favor_comment_df.iloc[0]['user_id']}, Score: {extreme_favor_user[stance]}")
            print_comments(extreme_favor_comment_df, stance)

            extreme_against_comment_df = dataset.get_user_comments(extreme_against_user['user'])
            print(f"Extreme Against User: {extreme_against_comment_df.iloc[0]['user_id']}, Score: {extreme_against_user[stance]}")
            print_comments(extreme_against_comment_df, stance)

            moderate_comment_df = dataset.get_user_comments(moderate_user['user'])
            print(f"Moderate User: {moderate_comment_df.iloc[0]['user_id']}, Score: {moderate_user[stance]}")
            print_comments(moderate_comment_df, stance)

if __name__ == '__main__':
    main()