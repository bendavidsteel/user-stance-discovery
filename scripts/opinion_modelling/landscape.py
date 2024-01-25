import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.neighbors import KernelDensity

import opinion_datasets, measures

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    model = sm.OLS(x, y)
    res = model.fit()
    corr = res.params.iloc[0]
    pval = res.pvalues.iloc[0]
    r2 = res.rsquared
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {corr:.2f}, p-val = {pval:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate(f'R² = {r2:.2f}', xy=(.1, .8), xycoords=ax.transAxes)

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
    min_time_step = dataset.min_time_step.to_pydatetime()
    max_time_step = dataset.max_time_step.to_pydatetime()

    snapshot = 'month'

    if snapshot == 'month':
        # get all months between min and max time step
        starts = []
        ends = []
        start_year = min_time_step.year
        start_month = min_time_step.month
        start = datetime.datetime(start_year, start_month, 1)
        while start < max_time_step:
            if start.month == 12:
                end = datetime.datetime(start.year + 1, 1, 1)
            else:
                end = datetime.datetime(start.year, start.month + 1, 1)
            starts.append(start)
            ends.append(end)
            start = end
    else:
        starts = [min_time_step]
        ends = [max_time_step]

    opinion_means, opinion_variances, users = dataset.get_data(start=starts, end=ends)

    pretty_stances = [stance.replace('stance_', '').replace('_', ' ').title() for stance in dataset.stance_columns]

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

    
    do_pairplot = True
    if do_pairplot and opinion_means.shape[0] == 1:
        num_bins = 25
        # plot pairplot from opinion means and variances
        user_opinion_df = pd.DataFrame(opinion_means[0], columns=pretty_stances)
        user_opinion_df["user"] = users
        g = sns.PairGrid(user_opinion_df, dropna=True)
        g.map_diag(sns.histplot, bins=num_bins)
        g.map_upper(sns.scatterplot)
        g.map_upper(corrfunc)
        g.map_lower(sns.kdeplot)
        g.savefig(os.path.join(root_dir_path, "figs", "opinion_pairplot.png"))

        height = len(dataset.stance_columns) * 2.5
        with sns.utils._disable_autolayout():
            fig = plt.figure(figsize=(height, height))
        
        axes = [[None for _ in range(len(dataset.stance_columns))] for _ in range(len(dataset.stance_columns))]
        for i, stance1 in enumerate(pretty_stances):
            for j, stance2 in enumerate(pretty_stances):
                if i == j:
                    share_ax = axes[i-1][j-1] if i > 0 else None
                    ax = fig.add_subplot(len(dataset.stance_columns), len(dataset.stance_columns), i * len(dataset.stance_columns) + j + 1, sharex=share_ax)
                    prob_density = sum_gaussians(opinion_means[0], opinion_variances[0], i, num_bins)
                    ax.plot(np.linspace(-1, 1, num_bins), prob_density)
                    if i == len(dataset.stance_columns) - 1:
                        ax.set_xlabel(stance1)
                    if j == 0:
                        ax.set_ylabel(stance1)
                else:
                    share_y_ax = axes[i][j-1] if j > 0 and j-1 != i else None
                    share_x_ax = axes[i-1][j] if i > 0 and i-1 != j else None
                    ax = fig.add_subplot(len(dataset.stance_columns), len(dataset.stance_columns), i * len(dataset.stance_columns) + j + 1, sharex=share_x_ax, sharey=share_y_ax)
                    prob_density = sum_multi_gaussians(opinion_means[0], opinion_variances[0], i, j, num_bins)
                    ax.imshow(prob_density, extent=[-1,1,-1,1], origin='lower', cmap='viridis')
                    if i == len(dataset.stance_columns) - 1:
                        ax.set_xlabel(stance2)
                    if j == 0:
                        ax.set_ylabel(stance1)
                axes[i][j] = ax
        fig.savefig(os.path.join(root_dir_path, "figs", "opinion_distribution_pairplot.png"))


    do_dist_change = True
    if do_dist_change and opinion_means.shape[0] > 1:
        num_bins = 25
        height = len(dataset.stance_columns) * 2.5
        with sns.utils._disable_autolayout():
            fig = plt.figure(figsize=(height, height))
        
        axes = [[None for _ in range(len(dataset.stance_columns))] for _ in range(len(dataset.stance_columns))]
        for i, stance1 in enumerate(pretty_stances):
            for j, stance2 in enumerate(pretty_stances):
                if i == j:
                    share_ax = axes[i-1][j-1] if i > 0 else None
                    ax = fig.add_subplot(len(dataset.stance_columns), len(dataset.stance_columns), i * len(dataset.stance_columns) + j + 1)
                    mean_timeline = np.zeros(opinion_means.shape[0])
                    var_timeline = np.zeros(opinion_means.shape[0])
                    for k in range(opinion_means.shape[0]):
                        # remove nans
                        stance_opinion_means = opinion_means[k, :, i]
                        stance_opinion_means = stance_opinion_means[~np.isnan(stance_opinion_means)]
                        hist, bin_edges = np.histogram(stance_opinion_means, bins=num_bins, density=False)
                        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                        hist_mean = np.sum(hist * bin_centres) / np.sum(hist)
                        hist_var = np.sum(hist * (bin_centres - hist_mean) ** 2) / (np.sum(hist) - 1)
                        mean_timeline[k] = hist_mean
                        var_timeline[k] = hist_var

                    ax.errorbar(ends, mean_timeline, yerr=np.sqrt(var_timeline))
                    ax.xaxis_date()

                    if i == len(dataset.stance_columns) - 1:
                        ax.set_xlabel(stance1)
                    if j == 0:
                        ax.set_ylabel(stance1)
                else:
                    share_y_ax = axes[i][j-1] if j > 0 and j-1 != i else None
                    share_x_ax = axes[i-1][j] if i > 0 and i-1 != j else None
                    ax = fig.add_subplot(len(dataset.stance_columns), len(dataset.stance_columns), i * len(dataset.stance_columns) + j + 1)
                    
                    corr_timeline = np.zeros(opinion_means.shape[0])
                    rsquared_timeline = np.zeros(opinion_means.shape[0])
                    for k in range(opinion_means.shape[0]):
                        stance_opinion_means = opinion_means[k, :, [i,j]]
                        stance_opinion_means = stance_opinion_means[:,~np.isnan(stance_opinion_means).any(axis=0)]
                        model = sm.OLS(stance_opinion_means[0], stance_opinion_means[1])
                        res = model.fit()
                        corr = res.params[0]
                        r2 = res.rsquared
                        corr_timeline[k] = corr
                        rsquared_timeline[k] = r2

                    ax.errorbar(ends, corr_timeline, yerr=1 - rsquared_timeline)
                    ax.xaxis_date()

                    if i == len(dataset.stance_columns) - 1:
                        ax.set_xlabel(stance2)
                    if j == 0:
                        ax.set_ylabel(stance1)
                axes[i][j] = ax

        fig.autofmt_xdate()
        fig.savefig(os.path.join(root_dir_path, "figs", "opinion_timeline_pairplot.png"))

    do_polarization = True
    if do_polarization:
        def prep_for_polarization(opinion_means, i=0):
            cols = [i for i, stance in enumerate(dataset.stance_columns)]
            opinion_means = opinion_means[i,:,cols].T
            opinion_means = np.nan_to_num(opinion_means)
            opinion_means = 0.5 * (opinion_means + 1)
            return opinion_means
        
        if opinion_means.shape[0] > 1:
            symmetric_polarization_timeline = np.zeros(opinion_means.shape[0])
            for i in range(opinion_means.shape[0]):
                polarization = measures.symmetric_polarization(prep_for_polarization(opinion_means, i))
                symmetric_polarization_timeline[i] = polarization
                asymm_polarizations = measures.asymmetric_polarization(prep_for_polarization(opinion_means, i))
            fig, ax = plt.subplots()
            ax.plot(ends, symmetric_polarization_timeline)
            ax.xaxis_date()
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Symmetric Polarization")
            fig.autofmt_xdate()
            fig.savefig(os.path.join(root_dir_path, "figs", "symmetric_polarization_timeline.png"))
            
        else:
            polarization = measures.symmetric_polarization(prep_for_polarization(opinion_means))
            partitions, asymm_polarizations = measures.asymmetric_polarization(prep_for_polarization(opinion_means))
            print(f"Symmetric Polarization: {polarization}")

            # get top 5 polarized partitions
            top_polarization = np.sort(asymm_polarizations)[-5:]
            top_polarization_indices = np.argsort(asymm_polarizations)[-5:]
            top_partitions = [partitions[i] for i in top_polarization_indices]
            for i in range(4, -1, -1):
                partition = top_partitions[i]
                partition_str = ", ".join([dataset.stance_columns[j].replace('stance_', '').replace('_', ' ').title() for j in partition])
                print(f"Partition {i+1}: {partition_str}, Asymmetric Polarization: {top_polarization[i]}")

            # get least polarized partitions
            top_polarization = np.sort(asymm_polarizations)[:5]
            top_polarization_indices = np.argsort(asymm_polarizations)[:5]
            top_partitions = [partitions[i] for i in top_polarization_indices]
            for i in range(5):
                partition = top_partitions[i]
                partition_str = ", ".join([dataset.stance_columns[j].replace('stance_', '').replace('_', ' ').title() for j in partition])
                print(f"Partition {i+1}: {partition_str}, Asymmetric Polarization: {top_polarization[i]}")

    get_examples = False
    if get_examples:
        # get examples of moderate and extreme users for each opinion
        for stance in dataset.stance_columns:
            user_stance_df = user_opinion_df[user_opinion_df[stance].notna()]
            extreme_favor_user = user_stance_df.sort_values(by=stance, ascending=False).head(1).iloc[0]
            extreme_against_user = user_stance_df.sort_values(by=stance, ascending=True).head(1).iloc[0]
            # get user with stance closest to 0
            moderate_user = user_stance_df.iloc[user_stance_df[stance].abs().argsort().iloc[0]]

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