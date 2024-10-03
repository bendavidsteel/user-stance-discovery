
import os

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from mining.datasets import RedditOpinionTimelineDataset
from mining.dynamics import flow_pairplot
from mining.estimate import prep_gp_data, train_gaussian_process, get_gp_means, get_splines, get_spline_means

from test_curve_fitters import plot_spline

def plot(
    model,
    likelihood,
    train_x=None,
    train_y=None,
    plot_observed_data=False,
    plot_predictions=False,
    n_prior_samples=0,
    n_test=500,
    ax=None,
):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Initialize plot
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

        if plot_observed_data:
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        if plot_predictions:
            min_x = torch.min(train_x)
            max_x = torch.max(train_x)
            start_x = min_x - 0.1 * (max_x - min_x)
            end_x = max_x + 0.1 * (max_x - min_x)
            test_x = torch.linspace(start_x, end_x, n_test)
            test_classifier_ids = torch.zeros(n_test, dtype=torch.long)
            batch_size = 10
            batches = [test_x[i:i+batch_size] for i in range(0, len(test_x), batch_size)]
            with gpytorch.settings.cholesky_max_tries(5):
                model_pred = model(batches[0])
                loc = model_pred.loc
                lower, upper = model_pred.confidence_region()
                for batch in batches[1:]:
                    model_pred = model(batch)
                    loc = torch.cat([loc, model_pred.loc], dim=0)
                    # Get upper and lower confidence bounds
                    batch_lower, batch_upper = model_pred.confidence_region()
                    lower = torch.cat([lower, batch_lower], dim=0)
                    upper = torch.cat([upper, batch_upper], dim=0)
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), loc.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    data_source = "reddit"
    source_dir = "1sub_1year"
    if source_dir == "4sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", data_source, source_dir)
    elif source_dir == "1sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", data_source, source_dir, "topics_minilm_0_2")
    subsample_users = None
    dataset = RedditOpinionTimelineDataset(topics_dir_path, subsample_users=subsample_users)

    all_classifier_profiles = dataset.all_classifier_profiles
    stance_targets = dataset.stance_columns
    X_norm, X, y, classifier_ids = prep_gp_data(dataset)

    model_type = 'spline'
    if model_type == 'gp':
        max_x = torch.max(torch.nan_to_num(X, 0))
        min_x = torch.min(torch.nan_to_num(X, torch.inf))
        max_x_norm = torch.max(torch.nan_to_num(X_norm, 0))
        min_x_norm = torch.min(torch.nan_to_num(X_norm, torch.inf))
        time_span_ratio = (max_x - min_x) / (max_x_norm - min_x_norm)
        month_span_ratio = time_span_ratio / (60 * 60 * 24 * 30)
        lengthscale_loc = month_span_ratio * 0.5
        lengthscale_scale = month_span_ratio * 0.1

        model_list, likelihood_list, model_map, losses = train_gaussian_process(X_norm, y, classifier_ids, stance_targets, all_classifier_profiles, lengthscale_loc=lengthscale_loc, lengthscale_scale=lengthscale_scale, gp_type='ordinal')
        timestamps, means, confidence_region = get_gp_means(dataset, model_list, likelihood_list, model_map, X_norm, X, y)
    elif model_type == 'spline':
        model_list, model_map = get_splines(X_norm, y, alpha=1e-1, n_knots=4)
        timestamps, means = get_spline_means(dataset, model_list, model_map, X_norm, X, y)

    np.save(os.path.join(root_dir_path, "data", data_source, source_dir, "timestamps.npy"), timestamps)
    np.save(os.path.join(root_dir_path, "data", data_source, source_dir, "means.npy"), means)
    if model_type == 'gp':
        np.save(os.path.join(root_dir_path, "data", data_source, source_dir, "confidence_region.npy"), confidence_region)

    os.makedirs(os.path.join(root_dir_path, "figs", data_source, source_dir, "model_fits"), exist_ok=True)
    for idx in tqdm.tqdm(range(min(len(model_list), 100)), desc="Plotting model fits"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
        i, k = model_map[idx]
        model = model_list[idx]
        train_x = X_norm[i, ~torch.isnan(y[i,:,k])]
        train_y = y[i, ~torch.isnan(y[i,:,k]), k]
        if model_type == 'gp':
            likelihood = likelihood_list[idx]
            plot(model, likelihood, train_x=train_x, train_y=train_y, plot_observed_data=True, plot_predictions=True, ax=ax)
        elif model_type == 'spline':
            plot_spline(model, train_x=train_x, train_y=train_y, plot_observed_data=True, plot_predictions=True, ax=ax)
        fig.savefig(os.path.join(root_dir_path, "figs", data_source, source_dir, "model_fits", f"{i}_{k}_gp.png"))
        plt.close(fig)


            
if __name__ == '__main__':
    main()

