
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
    # dataset = RedditOpinionTimelineDataset(topics_dir_path, subsample_users=subsample_users)

    model_type = 'spline'

    stance_columns = ['vaccine_mandates', 'renter_protections', 'ndp', 'liberals', 'conservatives', 'gun_control', 'drug_decriminalization', 'liberal_immigration_policy', 'canadian_aid_to_ukraine']
    timestamps = np.load(os.path.join(root_dir_path, "data", data_source, source_dir, "timestamps.npy"))
    means = np.load(os.path.join(root_dir_path, "data", data_source, source_dir, "means.npy"))
    if model_type == 'gp':
        confidence_region = np.load(os.path.join(root_dir_path, "data", data_source, source_dir, "confidence_region.npy"))


    flow_dir_path = os.path.join(root_dir_path, "figs", data_source, source_dir, "flows")
    os.makedirs(flow_dir_path, exist_ok=True)

    # TODO ensure timestamps are correctly plotted, the figure has suspiciously many people with similar start times
    figs, axes = flow_pairplot(timestamps, means, stance_columns, do_pairplot=False)

    for i in range(len(axes)):
        for j in range(len(axes[i])):
            fig = figs[i, j]
            fig.savefig(os.path.join(flow_dir_path, f'{i}_{j}_flows.png'))

            
if __name__ == '__main__':
    main()

