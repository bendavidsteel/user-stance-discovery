import datetime
import json
import os
from inspect import signature

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

import opinion_datasets, measures

def weighted_pearson(x, x_var, y, y_var):
    x_weights = np.exp(-x_var)
    x_weights = x_weights / len(x)
    y_weights = np.exp(-y_var)
    y_weights = y_weights / len(y)
    x_mean = np.average(x, weights=x_weights)
    y_mean = np.average(y, weights=y_weights)
    x_var = np.average((x - x_mean) ** 2, weights=x_weights)
    y_var = np.average((y - y_mean) ** 2, weights=y_weights)
    cov = np.average((x - x_mean) * (y - y_mean), weights=x_weights * y_weights)
    pearson = cov / np.sqrt(x_var * y_var)
    r_squared = pearson ** 2
    t_weighted = pearson * np.sqrt((np.sum(x_weights * y_weights) - 2) / (1 - r_squared))
    pval = 2 * (1 - scipy.stats.t.cdf(np.abs(t_weighted), len(x) - 2))
    return pearson, r_squared, pval

def descr_stats_w(x, x_var, y, y_var):
    weights = np.exp(-np.nanmean(np.stack([x_var, y_var], axis=-1), axis=1))
    weights = weights / np.mean(weights)
    data = np.stack([x, y], axis=-1)
    model = DescrStatsW(data, weights=weights)
    corrcoef = model.corrcoef
    r_squared = corrcoef ** 2
    tstat, pval, df = model.ttest_mean(0)
    return corrcoef, r_squared, pval

def weighted_least_squares(x, x_var, y, y_var):
    # calculate confidence as 1 / (mean of variances)
    eps = 1e-6
    confidence = 1 / np.maximum(((x_var + y_var) / 2), 0.01)

    # sort values so that fitted line is already sorted
    x_sort_args = np.argsort(x)
    x = x[x_sort_args]
    y = y[x_sort_args]

    model = sm.WLS(y, sm.add_constant(x), weights=confidence)
    res = model.fit()
    const = res.params[0]
    corr = res.params[1]
    pval = res.pvalues[1]
    r2 = res.rsquared
    return corr, r2, pval, const

def ax_corrfunc(ax, x_var, y_var, df):
    if ax is None:  # i.e. we are in corner mode
        return

    if x_var == y_var:
        axes_vars = [x_var]
    else:
        axes_vars = [x_var, y_var]

    data = df[axes_vars + [f"{x_var} variance", f"{y_var} variance"]]
    data = data.dropna()

    x = data[x_var].values
    y = data[y_var].values
    x_var_variance = data[f"{x_var} variance"].values
    y_var_variance = data[f"{y_var} variance"].values

    if len(x) < 2:
        return        

    corr, r2, pval, const = weighted_least_squares(x, x_var_variance, y, y_var_variance)

    d = np.linspace(-1, 1, 100)
    fit = const + d * corr
    ax.plot(d, fit, color='r', linestyle='--', linewidth=2)

    ax = ax or plt.gca()
    ax.annotate(f'ρ = {corr:.2f}, p-val = {pval:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate(f'R² = {r2:.2f}', xy=(.1, .8), xycoords=ax.transAxes)

def corrfunc(pairgrid: sns.PairGrid):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    indices = zip(*np.triu_indices_from(pairgrid.axes, 1))
    for i, j in indices:
        x_var = pairgrid.x_vars[j]
        y_var = pairgrid.y_vars[i]
        ax = pairgrid.axes[i, j]
        ax_corrfunc(ax, x_var, y_var, pairgrid.data)

def plot_beta(x, **kws):
    x_var = x.name
    df = kws['df']
    data = df[[f"{x_var} beta alpha", f"{x_var} beta beta"]].dropna().values
    epsilon = 1e-6
    beta_domain = np.linspace(0 + epsilon, 1 - epsilon, 100)
    y = np.zeros_like(beta_domain)
    for i in range(data.shape[0]):
        alpha, beta = data[i]
        y += scipy.stats.beta.pdf(beta_domain, alpha, beta) / data.shape[0]

    stance_domain = np.linspace(-1, 1, 100)
    plt.plot(stance_domain, y)
    plt.xlim(-1, 1)

    # calculate mean and plot it 
    hist_mean = np.sum(y * stance_domain) / np.sum(y)
    plt.axvline(hist_mean, color='r', linestyle='--')
    plt.annotate(f"Mean: {hist_mean:.2f}", xy=(.1, .9), xycoords='axes fraction')

def sum_bivariate_betas(x, y, **kws):
    x_var = x.name
    y_var = y.name
    df = kws['df']
    data = df[[f"{x_var} beta alpha", f"{x_var} beta beta", f"{y_var} beta alpha", f"{y_var} beta beta"]].dropna().values
    x_data = data[:,:2]
    y_data = data[:,2:]
    epsilon = 1e-6
    d = np.linspace(0 + epsilon, 1 - epsilon, 100)
    pdf = np.zeros((100, 100))
    for i in range(x_data.shape[0]):
        x_alpha, x_beta = x_data[i]
        y_alpha, y_beta = y_data[i]
        x_pdf = scipy.stats.beta.pdf(d, x_alpha, x_beta)
        y_pdf = scipy.stats.beta.pdf(d, y_alpha, y_beta)
        pdf += np.outer(x_pdf, y_pdf) / x_data.shape[0]

    plt.contour(pdf, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    # plt.matshow(pdf, cmap='viridis', extent=[-1, 1, -1, 1], origin='lower')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

def plot_gaussians(x, **kws):
    x_var = x.name
    df = kws['df']
    data = df[[f"{x_var} inferred mean", f"{x_var} inferred variance"]].dropna().values
    y = sum_gaussians(data[:,0], data[:,1], 100)

    stance_domain = np.linspace(-1, 1, 100)
    plt.plot(stance_domain, y)
    plt.xlim(-1, 1)

    # calculate mean and plot it 
    hist_mean = np.sum(y * stance_domain) / np.sum(y)
    plt.axvline(hist_mean, color='r', linestyle='--')
    plt.annotate(f"Mean: {hist_mean:.2f}", xy=(.1, .9), xycoords='axes fraction')

def plot_bivariate_gaussians(x, y, **kws):
    x_var = x.name
    y_var = y.name
    df = kws['df']
    data = df[[f"{x_var} inferred mean", f"{x_var} inferred variance", f"{y_var} inferred mean", f"{y_var} inferred variance"]].dropna().values
    x_data = data[:,:2]
    y_data = data[:,2:]
    y = sum_multi_gaussians(data[:,[0, 2]], data[:,[1, 3]], 100)

    plt.contour(y, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    # plt.matshow(pdf, cmap='viridis', extent=[-1, 1, -1, 1], origin='lower')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

def sum_gaussians(opinion_means, opinion_variances, num_bins):
    # Create a linspace array which will be reused
    linspace_array = np.linspace(-1, 1, num_bins)

    # Identify valid indices where none of the values are NaN
    valid_indices = ~np.isnan(opinion_means) & ~np.isnan(opinion_variances) & (opinion_variances > 0)

    # Extract the valid means and variances
    valid_means = opinion_means[valid_indices]
    valid_variances = opinion_variances[valid_indices]

    # Calculate the probability densities for the valid pairs
    prob_density = 1 / (valid_variances * np.sqrt(2 * np.pi)) * np.exp(-(linspace_array[:, None] - valid_means) ** 2 / (2 * valid_variances))
    # normalize each users normal distribution
    prob_density = prob_density / np.sum(prob_density, axis=-2, keepdims=True)
    # sum all the users distributions
    prob_density = np.sum(prob_density, axis=-1)

    # add point masses
    valid_indices = ~np.isnan(opinion_means) & ~np.isnan(opinion_variances) & (opinion_variances == 0)
    valid_means = opinion_means[valid_indices]
    for mean in valid_means:
        prob_density[np.argmin(np.abs(linspace_array - mean))] += 1

    # normalize distribution
    prob_density /= np.sum(prob_density, axis=-1)
    return prob_density

def sum_multi_gaussians(opinion_means, opinion_variances, num_bins):
    # ensure the bivariate dimension is last
    if opinion_means.shape[0] == 2:
        opinion_means = opinion_means.T
        opinion_variances = opinion_variances.T

    assert opinion_means.shape[-1] == 2
    assert opinion_variances.shape[-1] == 2

    # Create a linspace array which will be reused
    positions = np.stack(np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)), axis=-1)

    # Identify valid indices where none of the values are NaN
    valid_indices = ~np.isnan(opinion_means[:,0]) & ~np.isnan(opinion_variances[:,0]) & ~np.isnan(opinion_means[:,1]) & ~np.isnan(opinion_variances[:,1]) & (opinion_variances[:,0] > 0) & (opinion_variances[:,1] > 0)

    # Calculate the probability densities for the valid pairs
    dists = positions - opinion_means[valid_indices,:].reshape(-1,1,1,2)
    dists = np.reshape(dists, (dists.shape[:-1] + (2, 1)))
    expanded_variance = opinion_variances[valid_indices,:].reshape(-1,1,1,2)
    cov_mat = np.zeros((expanded_variance.shape[0], 1, 1, 2, 2))
    cov_mat[..., 0, 0] = expanded_variance[..., 0]
    cov_mat[..., 1, 1] = expanded_variance[..., 1]
    prob_density = (1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov_mat))).reshape(-1, 1, 1, 1, 1) * np.exp(-0.5 * np.transpose(dists, (0,1,2,4,3)) @ np.linalg.inv(cov_mat) @ dists)

    # Sum over the second dimension to get the total probability density
    prob_density = prob_density.reshape(-1, 25, 25)
    prob_density = prob_density / np.sum(prob_density, axis=(1,2), keepdims=True)
    prob_density = np.sum(prob_density, axis=0).reshape(25, 25)

    # add point masses
    valid_indices = ~np.isnan(opinion_means[:,0]) & ~np.isnan(opinion_variances[:,0]) & ~np.isnan(opinion_means[:,1]) & ~np.isnan(opinion_variances[:,1]) & (opinion_variances[:,0] == 0) & (opinion_variances[:,1] == 0)
    valid_means = opinion_means[valid_indices,:]
    for idx in range(valid_means.shape[0]):
        prob_density[np.argmin(np.abs(positions[0,:,0] - valid_means[idx,0])), np.argmin(np.abs(positions[:,0,1] - valid_means[idx,1]))] += 1

    prob_density /= np.sum(prob_density)
    return prob_density

def plot_pairplot(opinion_means, opinion_variances, opinion_normal, users, pretty_stances, fig_path, subreddit, num_bins=25):
    opinion_inferred_mean, opinion_inferred_variance = opinion_normal

    # user_opinion_df = pd.DataFrame(
    #     np.concatenate([opinion_means[0], opinion_variances[0], opinion_inferred_mean[0], opinion_inferred_variance[0]], axis=1),
    #     columns=pretty_stances + [f"{stance} variance" for stance in pretty_stances] + [f"{stance} inferred mean" for stance in pretty_stances] + [f"{stance} inferred variance" for stance in pretty_stances]
    # )
    opinion_inferred_mean[opinion_means == np.nan] = np.nan
    user_opinion_df = pd.DataFrame(
        np.concatenate([opinion_inferred_mean[0], opinion_inferred_variance[0], opinion_inferred_mean[0], opinion_inferred_variance[0]], axis=1),
        columns=pretty_stances + [f"{stance} variance" for stance in pretty_stances] + [f"{stance} inferred mean" for stance in pretty_stances] + [f"{stance} inferred variance" for stance in pretty_stances]
    )
    user_opinion_df[users.columns] = users
    if subreddit != 'all':
        user_opinion_df = user_opinion_df[user_opinion_df['subreddit'] == subreddit]
    
    matplotlib.rcParams["axes.labelsize"] = 12
    g = sns.PairGrid(user_opinion_df, vars=pretty_stances, dropna=True)
    g.map_diag(plot_gaussians, df=user_opinion_df)
    g.map_upper(sns.scatterplot)
    g.apply(corrfunc)
    g.map_lower(plot_bivariate_gaussians, df=user_opinion_df)
    g.savefig(os.path.join(fig_path, "opinion_pairplot.png" if subreddit == 'all' else f"opinion_pairplot_{subreddit}.png"))

    if True:
        if subreddit != 'all':
            opinion_means = opinion_means[0, users['subreddit'] == subreddit,:]
            opinion_variances = opinion_variances[0, users['subreddit'] == subreddit,:]

        scatter_path = os.path.join(fig_path, 'scatters')
        os.makedirs(scatter_path, exist_ok=True)
        
        axes = [[None for _ in range(len(pretty_stances))] for _ in range(len(pretty_stances))]
        for i, stance1 in enumerate(pretty_stances):
            for j, stance2 in enumerate(pretty_stances):
                if j > i:
                    fig, ax = plt.subplots(figsize=(3, 2.8))
                    sns.scatterplot(data=user_opinion_df, x=stance1, y=stance2, ax=ax)
                    ax_corrfunc(ax, stance2, stance1, user_opinion_df)
                    ax.set_xlabel(stance2)
                    ax.set_ylabel(stance1)
                    fig.tight_layout()
                    fig.savefig(os.path.join(scatter_path, f"{stance1}_{stance2}_scatter.png" if subreddit == 'all' else f"{stance1}_{stance2}_scatter_{subreddit}.png"))
                    plt.close(fig)

                    # remove users with zero user stance and recalculate correlation
                    valid_indices = ~np.isnan(opinion_inferred_mean[0,:,i]) & ~np.isnan(opinion_inferred_mean[0,:,j]) & (opinion_inferred_mean[0,:,i] != 0) & (opinion_inferred_mean[0,:,j] != 0)
                    x = opinion_inferred_mean[0,valid_indices,j]
                    y = opinion_inferred_mean[0,valid_indices,i]
                    x_var_variance = opinion_inferred_variance[0,valid_indices,j]
                    y_var_variance = opinion_inferred_variance[0,valid_indices,i]
                    corr, r2, pval, const = weighted_least_squares(x, x_var_variance, y, y_var_variance)
                    print(f"{stance1} and {stance2} correlation: {corr:.2f}, r2: {r2:.2f}, pval: {pval:.2f}")

def plot_pairplot_categorical(opinion_categorical, users, pretty_stances, fig_path, subreddit):

    height = len(pretty_stances) * 2.5
    with sns.utils._disable_autolayout():
        fig = plt.figure(figsize=(height, height))

    if subreddit != 'all':
        opinion_categorical = opinion_categorical[0, users['subreddit'] == subreddit,:]
    
    axes = [[None for _ in range(len(pretty_stances))] for _ in range(len(pretty_stances))]
    for i, stance1 in enumerate(pretty_stances):
        for j, stance2 in enumerate(pretty_stances):
            if i == j:
                share_ax = axes[i-1][j-1] if i > 0 else None
                ax = fig.add_subplot(len(pretty_stances), len(pretty_stances), i * len(pretty_stances) + j + 1, sharex=share_ax)
                prob_density = np.mean(opinion_categorical[0,:,i], axis=0)
                ax.bar(np.linspace(-1, 1, prob_density.shape[-1]), prob_density)
                if i == len(pretty_stances) - 1:
                    ax.set_xlabel(stance1)
                if j == 0:
                    ax.set_ylabel(stance1)
            else:
                share_y_ax = axes[i][j-1] if j > 0 and j-1 != i else None
                share_x_ax = axes[i-1][j] if i > 0 and i-1 != j else None
                ax = fig.add_subplot(len(pretty_stances), len(pretty_stances), i * len(pretty_stances) + j + 1, sharex=share_x_ax, sharey=share_y_ax)
                prob_density = np.mean(np.einsum('ni,nj->nij', opinion_categorical[0,:,i], opinion_categorical[0,:,j]), axis=0)
                ax.imshow(prob_density, extent=[-1,1,-1,1], origin='lower', cmap='viridis')
                if i == len(pretty_stances) - 1:
                    ax.set_xlabel(stance2)
                if j == 0:
                    ax.set_ylabel(stance1)
            axes[i][j] = ax
    fig.savefig(os.path.join(fig_path, "opinion_categorical_pairplot.png" if subreddit == 'all' else f"opinion_categorical_pairplot_{subreddit}.png"))


def plot_dist_change(opinion_means, opinion_variances, dataset, pretty_stances, fig_path, ends):
    num_bins = 25
    error_method = 'conf'
    if not os.path.exists(os.path.join(fig_path, 'timelines')):
        os.makedirs(os.path.join(fig_path, 'timelines'))
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
                error_timeline = np.zeros((opinion_means.shape[0], 2))
                for k in range(opinion_means.shape[0]):
                    # remove nans
                    stance_opinion_means = opinion_means[k, :, i]
                    stance_opinion_vars = opinion_variances[k, :, i]
                    valid_indices = ~np.isnan(stance_opinion_means) & ~np.isnan(stance_opinion_vars)
                    stance_opinion_means = stance_opinion_means[valid_indices]
                    stance_opinion_vars = stance_opinion_vars[valid_indices]

                    if stance_opinion_means.shape[0] < 2:
                        continue

                    if error_method == 'hist':
                        hist = sum_betas()
                        hist, bin_edges = np.histogram(stance_opinion_means, bins=num_bins, density=False)
                        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                        hist_mean = np.sum(hist * bin_centres) / np.sum(hist)
                        hist_var = np.sum(hist * (bin_centres - hist_mean) ** 2) / (np.sum(hist) - 1)
                        mean_timeline[k] = hist_mean
                        error_timeline[k, :] = np.sqrt(hist_var)
                    elif error_method == 'conf':
                        # def get_hist_mean(mean, var, axis=0):
                        #     # mean = data[...,0]
                        #     # var = data[...,1]
                        #     pdf = sum_gaussians(mean, var, num_bins)
                        #     hist_mean = np.sum(pdf * np.linspace(-1, 1, num_bins), axis=-1) / np.sum(pdf, axis=-1)
                        #     return hist_mean

                        # stance_opinion_mv = np.stack([stance_opinion_means, stance_opinion_vars], axis=-1)
                        # mean_timeline[k] = get_hist_mean(stance_opinion_means, stance_opinion_vars)

                        def get_weighted_mean(mean, var, axis=0):
                            weights = np.exp(-var)
                            weights = weights / np.mean(weights, axis=axis, keepdims=True)
                            hist_mean = np.average(mean, weights=weights, axis=axis)
                            return hist_mean

                        mean_timeline[k] = get_weighted_mean(stance_opinion_means, stance_opinion_vars, axis=0)
                        res = scipy.stats.bootstrap((stance_opinion_means, stance_opinion_vars), get_weighted_mean, vectorized=True, paired=True)
                        error_timeline[k, 0] = res.confidence_interval.low
                        error_timeline[k, 1] = res.confidence_interval.high

                solo_fig, solo_ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 2))

                if error_method == 'hist':
                    ax.errorbar(ends, mean_timeline, yerr=error_timeline)
                    solo_ax.errorbar(ends, mean_timeline, yerr=error_timeline)
                elif error_method == 'conf':
                    ax.plot(ends, mean_timeline)
                    ax.fill_between(ends, error_timeline[:, 0], error_timeline[:, 1], alpha=0.3)
                    solo_ax.plot(ends, mean_timeline)
                    solo_ax.fill_between(ends, error_timeline[:, 0], error_timeline[:, 1], alpha=0.3)
                ax.xaxis_date()
                solo_ax.xaxis_date()

                if i == len(dataset.stance_columns) - 1:
                    ax.set_xlabel(stance1)
                if j == 0:
                    ax.set_ylabel(stance1)

                solo_ax.set_xlabel("Date")
                solo_ax.set_ylabel("Mean Opinion")
                # solo_ax.set_title(f"{stance1} Opinion Over Time")
                # solo_ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator((0,4,8,12)))
                # solo_ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((2,6,10)))

                solo_ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
                # solo_ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
                solo_fig.autofmt_xdate()
                solo_fig.tight_layout()
                solo_fig.savefig(os.path.join(fig_path, 'timelines', f"{stance1.lower().replace(' ', '_')}_opinion_timeline.png"))
                plt.close(fig)
            else:
                share_y_ax = axes[i][j-1] if j > 0 and j-1 != i else None
                share_x_ax = axes[i-1][j] if i > 0 and i-1 != j else None
                ax = fig.add_subplot(len(dataset.stance_columns), len(dataset.stance_columns), i * len(dataset.stance_columns) + j + 1)
                
                corr_timeline = np.zeros(opinion_means.shape[0])
                rsquared_timeline = np.zeros((opinion_means.shape[0], 2))
                for k in range(opinion_means.shape[0]):
                    stance_opinion_means = opinion_means[k, :, [i,j]]
                    stance_opinion_vars = opinion_variances[k, :, [i,j]]
                    valid_indices = ~np.isnan(stance_opinion_means).any(axis=0) & ~np.isnan(stance_opinion_vars).any(axis=0)
                    stance_opinion_means = stance_opinion_means[:,valid_indices]
                    stance_opinion_vars = stance_opinion_vars[:,valid_indices]

                    if stance_opinion_means.shape[1] < 2:
                        continue

                    if error_method == 'hist':

                        corr, r2, pval, const = weighted_least_squares(stance_opinion_means[0], stance_opinion_vars[0], stance_opinion_means[1], stance_opinion_vars[1])
                        corr_timeline[k] = corr
                        rsquared_timeline[k] = r2
                    elif error_method == 'conf':
                        def calc_corr(x, x_var, y, y_var, axis=0):
                            # x = x_mv[...,0]
                            # x_var = x_mv[...,1]
                            # y = y_mv[...,0]
                            # y_var = y_mv[...,1]
                            weights = np.exp(-np.nanmean(np.stack([x_var, y_var], axis=-1), axis=-1))
                            x_mean = np.average(x, weights=weights, axis=-1)
                            xv = x - x.mean(axis=-1, keepdims=True)
                            yv = y - y.mean(axis=-1, keepdims=True)
                            xvss = (xv * xv).sum(axis=-1)
                            yvss = (yv * yv).sum(axis=-1)
                            result = (xv * yv).sum(axis=-1) / (np.sqrt(xvss) * np.sqrt(yvss))
                            # bound the values to -1 to 1 in the event of precision issues
                            return np.maximum(np.minimum(result, 1.0), -1.0)
                            
                        x_mv = np.stack([stance_opinion_means[0], stance_opinion_vars[0]], axis=-1)
                        y_mv = np.stack([stance_opinion_means[1], stance_opinion_vars[1]], axis=-1)
                        corr_timeline[k] = calc_corr(stance_opinion_means[0], stance_opinion_vars[0], stance_opinion_means[1], stance_opinion_vars[1])
                        res = scipy.stats.bootstrap((stance_opinion_means[0], stance_opinion_vars[0], stance_opinion_means[1], stance_opinion_vars[1]), calc_corr, vectorized=True, paired=True)
                        rsquared_timeline[k, 0] = res.confidence_interval.low
                        rsquared_timeline[k, 1] = res.confidence_interval.high

                solo_fig, solo_ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 2))

                if error_method == 'hist':
                    ax.errorbar(ends, corr_timeline, yerr=1 - rsquared_timeline)
                    solo_ax.errorbar(ends, corr_timeline, yerr=1 - rsquared_timeline)
                elif error_method == 'conf':
                    ax.plot(ends, corr_timeline)
                    ax.fill_between(ends, rsquared_timeline[:, 0], rsquared_timeline[:, 1], alpha=0.3)
                    solo_ax.plot(ends, corr_timeline)
                    solo_ax.fill_between(ends, rsquared_timeline[:, 0], rsquared_timeline[:, 1], alpha=0.3)
                ax.xaxis_date()
                solo_ax.xaxis_date()

                if i == len(dataset.stance_columns) - 1:
                    ax.set_xlabel(stance2)
                if j == 0:
                    ax.set_ylabel(stance1)

                solo_ax.set_xlabel("Date")
                solo_ax.set_ylabel("Correlation")
                # solo_ax.set_title(f"{stance1} and {stance2} Correlation Over Time")
                # solo_ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
                # solo_ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((0,3,6,9)))

                solo_ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
                # solo_ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
                solo_fig.autofmt_xdate()
                solo_fig.tight_layout()
                solo_fig.savefig(os.path.join(fig_path, 'timelines', f"{stance1.lower().replace(' ', '_')}_{stance2.lower().replace(' ', '_')}_correlation_timeline.png"))
                plt.close(solo_fig)

            axes[i][j] = ax

    fig.autofmt_xdate()
    fig.savefig(os.path.join(fig_path, "opinion_timeline_pairplot.png"))

def calc_polarization(opinion_means, dataset, fig_path, ends):
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
        fig.savefig(os.path.join(fig_path, "symmetric_polarization_timeline.png"))
        
    else:
        polarization = measures.symmetric_polarization(prep_for_polarization(opinion_means))
        partitions, asymm_polarizations = measures.asymmetric_polarization(prep_for_polarization(opinion_means))
        print(f"Symmetric Polarization: {polarization}")

        # get top 5 polarized partitions
        top_polarization = np.sort(asymm_polarizations)[-5:]
        top_polarization_indices = np.argsort(asymm_polarizations)[-5:]
        top_partitions = [partitions[i] for i in top_polarization_indices]
        for i in range(min(len(dataset.stance_columns)-1, 4), -1, -1):
            partition = top_partitions[i]
            partition_str = ", ".join([dataset.stance_columns[j].replace('stance_', '').replace('_', ' ').title() for j in partition])
            print(f"Partition {i+1}: {partition_str}, Asymmetric Polarization: {top_polarization[i]}")

        # get least polarized partitions
        top_polarization = np.sort(asymm_polarizations)[:5]
        top_polarization_indices = np.argsort(asymm_polarizations)[:5]
        top_partitions = [partitions[i] for i in top_polarization_indices]
        for i in range(min(len(dataset.stance_columns)-1, 5)):
            partition = top_partitions[i]
            partition_str = ", ".join([dataset.stance_columns[j].replace('stance_', '').replace('_', ' ').title() for j in partition])
            print(f"Partition {i+1}: {partition_str}, Asymmetric Polarization: {top_polarization[i]}")

def cached_get_data(dataset, dataset_name, topics_dir_path, starts, ends, snapshot, subsample_users):
    cache_name = f"{dataset_name}_{dataset.aggregation}_{snapshot}"
    if subsample_users:
        cache_name += f"_subsample_{subsample_users}"
    mean_file_path = os.path.join(topics_dir_path, f"{cache_name}_mean.npy")
    variance_file_path = os.path.join(topics_dir_path, f"{cache_name}_variance.npy")
    users_file_path = os.path.join(topics_dir_path, f"{cache_name}_users.parquet.gzip")
    if os.path.exists(mean_file_path) and os.path.exists(variance_file_path) and os.path.exists(users_file_path):
        opinion_stats = (np.load(mean_file_path), np.load(variance_file_path))
        users = pd.read_parquet(users_file_path)
    else:
        opinion_stats, users = dataset.get_data(start=starts, end=ends)
        np.save(os.path.join(topics_dir_path, f"{cache_name}_mean.npy"), opinion_stats[0])
        np.save(os.path.join(topics_dir_path, f"{cache_name}_variance.npy"), opinion_stats[1])
        users.to_parquet(os.path.join(topics_dir_path, f"{cache_name}_users.parquet.gzip"), compression='gzip')

    return opinion_stats, users

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    dataset_name = "reddit"
    aggregation = "weighted_mean"
    snapshot = 'year'

    if dataset_name == "reddit":
        this_dir_path = os.path.dirname(os.path.realpath(__file__))
        root_dir_path = os.path.join(this_dir_path, "..", "..")
        experi = "4sub_1year"
        subsample_users = None
        if experi == "1sub_1year":
            topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "1sub_1year", "topics_minilm_0_2")
            fig_path = os.path.join(root_dir_path, "figs", "reddit", "1sub_1year")
        elif experi == "4sub_1year":
            topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "4sub_1year")
            fig_path = os.path.join(root_dir_path, "figs", "reddit", "4sub_1year")
        dataset = opinion_datasets.RedditOpinionTimelineDataset(topics_dir_path, aggregation=aggregation, subsample_users=subsample_users)

        min_time_step = dataset.min_time_step.to_pydatetime()
        max_time_step = dataset.max_time_step.to_pydatetime()

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

    elif dataset_name == "generative":
        num_people = 10
        num_opinions = 2
        num_data_points = 10
        user_stance = np.tile(np.clip(np.random.normal(0, 0.5, (num_people, num_opinions, 1)), -1, 1), num_data_points)
        user_stance_variance = np.tile(np.random.uniform(0, 0.5, (num_people, num_opinions, 1)), num_data_points)
        fig_path = os.path.join(root_dir_path, "figs", "generative")
        dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type='low_recall', aggregation=aggregation)

        if snapshot == 'year':
            starts = [0]
            ends = [num_data_points]
        elif snapshot == 'month':
            starts = []
            ends = []
            num_intervals = 2
            interval_size = num_data_points // num_intervals
            for i in range(0,num_data_points-interval_size+1, interval_size):
                starts.append(i)
                ends.append(i+interval_size)

    num_people = dataset.num_people

    if dataset_name == "reddit":
        opinion_stats, users = cached_get_data(dataset, dataset_name, topics_dir_path, starts, ends, snapshot, subsample_users)
    else:
        opinion_stats, users = dataset.get_data(start=starts, end=ends)
        
    # TODO plot histogram as sum of beta distributions
    # use weighted mean for scatter graphs
    # use sum of beta distributions instead of KDE plots

    pretty_stances = [stance.replace('stance_', '').replace('_', ' ').title() for stance in dataset.stance_columns]
    mapping = {'Ndp': 'NDP'}
    pretty_stances = [mapping[s] if s in mapping else s for s in pretty_stances]
    
    do_pairplot = True
    if do_pairplot and opinion_stats[0].shape[0] == 1:
        dataset.aggregation = "inferred_normal"
        opinion_stats_normal, users = cached_get_data(dataset, dataset_name, topics_dir_path, starts, ends, snapshot, subsample_users)

        if 'subreddit' in users.columns:
            subreddits = users['subreddit'].unique()
        else:
            subreddits = ['all']

        # plot pairplot from opinion means and variances
        if aggregation != "inferred_categorical":
            num_bins = 25
            opinion_means, opinion_variances = opinion_stats
            plot_pairplot(opinion_means, opinion_variances, opinion_stats_normal, users, pretty_stances, fig_path, 'all', num_bins)
            # for subreddit in subreddits:
            #     plot_pairplot(opinion_means, opinion_variances, opinion_stats_normal, users, pretty_stances, fig_path, subreddit, num_bins)
        elif aggregation == "inferred_categorical":
            opinion_categorical = opinion_stats[0]
            plot_pairplot_categorical(opinion_categorical, users, pretty_stances, fig_path, 'all')
            for subreddit in subreddits:
                plot_pairplot_categorical(opinion_categorical, users, pretty_stances, fig_path, subreddit)
            


    do_dist_change = True
    if do_dist_change and opinion_stats[0].shape[0] > 1:
        if aggregation != "inferred_categorical":
            opinion_means, opinion_variances = opinion_stats
            plot_dist_change(opinion_means, opinion_variances, dataset, pretty_stances, fig_path, ends)

    do_polarization = True
    if do_polarization:
        if 'subreddit' in users.columns:
            subreddits = users['subreddit'].unique()
        else:
            subreddits = ['all']

        if aggregation != "inferred_categorical":
            opinion_means, opinion_variances = opinion_stats
            print(f"All Subreddits")
            calc_polarization(opinion_means, dataset, fig_path, ends)
            for subreddit in subreddits:
                print(f"Subreddit: {subreddit}")
                opinion_means_subreddit = opinion_means[:, users['subreddit'] == subreddit,:]
                calc_polarization(opinion_means_subreddit, dataset, fig_path, ends)

    get_examples = False
    if get_examples:
        if aggregation == "inferred_categorical":
            raise NotImplementedError("Getting examples for inferred categorical opinions is not implemented")
        opinion_means = opinion_stats[0]
        user_opinion_df = pd.DataFrame(opinion_means[0], columns=pretty_stances)
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