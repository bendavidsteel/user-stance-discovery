
import os

import matplotlib.pyplot as plt

from mining.datasets import RedditOpinionTimelineDataset
from mining.dynamics import flow_pairplot
from mining.estimate import prep_gp_data, train_gaussian_process, get_gp_means


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    data_source = "reddit"
    source_dir = "1sub_1year"
    if source_dir == "4sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "4sub_1year")
    elif source_dir == "1sub_1year":
        topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "1sub_1year", "topics_minilm_0_2")
    subsample_users = 100
    dataset = RedditOpinionTimelineDataset(topics_dir_path, subsample_users=subsample_users)

    X_norm, X, y = prep_gp_data(dataset)
    model_list, likelihood_list, model_map, losses = train_gaussian_process(X_norm, y)
    timestamps, means = get_gp_means(dataset, model_list, likelihood_list, model_map, X_norm, X, y)

    flow_dir_path = os.path.join(root_dir_path, "figs", data_source, source_dir, "flows")
    os.makedirs(flow_dir_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")  # supress output text
    fig.savefig(os.path.join(flow_dir_path, 'losses.png'))

    # TODO ensure timestamps are correctly plotted, the figure has suspiciously many people with similar start times
    figs, axes = flow_pairplot(timestamps, means, dataset.stance_columns, do_pairplot=False)

    for i in range(len(axes)):
        for j in range(len(axes[i])):
            fig = figs[i, j]
            fig.savefig(os.path.join(flow_dir_path, f'{i}_{j}_flows.png'))

            
if __name__ == '__main__':
    main()

