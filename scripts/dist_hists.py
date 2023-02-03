import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EPSILON = 1e-6
MAX_EXP = 100
NUM_BINS = 100

def between_metric(other_user_dist, prev_other_dist, user_prev_dist):
    # if prev_other_dist != 0:
    #     return np.clip((user_prev_dist / prev_other_dist) * (user_prev_dist**2 + prev_other_dist**2 - other_user_dist**2) / (2 * user_prev_dist * prev_other_dist), 0, 1)
    # else:
    #     return 0
    return 1 / (1 + (np.e ** (other_user_dist - user_prev_dist)))

def away_metric(other_user_dist, prev_other_dist, user_prev_dist):
    if prev_other_dist != 0:
        dist_from_line = ((other_user_dist + user_prev_dist) / prev_other_dist) - 1
    else:
        dist_from_line = other_user_dist
    dist_from_line = min(dist_from_line, MAX_EXP)
    return np.e ** (dist_from_line) / (np.e ** (dist_from_line) + 1)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    num_dims = 5
    change_type = 'changepoint'

    file_name = f'user_seqs_{change_type}_{num_dims}.json'
    user_seq_path = os.path.join(data_dir_path, file_name)
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    dists = collections.defaultdict(list)

    edge_type = 'comment_reply'
    fig_type = 'movement'

    num_interactions = 0
    for user_id, user_seq in tqdm(user_seqs.items()):
        for inter in user_seq:
            if len(inter) > 3:
                num_interactions += 1
            inter_type = inter['edge_data']['type']
            for key, value in inter.items():
                if 'euclid' in key or 'cosine' in key:
                    dists[f"{key}_{inter_type}"].append(value)
                    if fig_type == 'movement' and '_user_' in key and 'prev' not in key:
                        key_suffix = 19 if 'euclid' in key else 18
                        other_user_dist = value
                        prev_other_dist = inter[key.replace('user', 'prev')]
                        user_prev_dist = inter[f"prev_user_{key[-key_suffix:]}"]
                        dists[f"{key}_between_{inter_type}"].append(between_metric(other_user_dist, prev_other_dist, user_prev_dist))
                        dists[f"{key}_away_{inter_type}"].append(away_metric(other_user_dist, prev_other_dist, user_prev_dist))
    print(f"Num interactions: {num_interactions}")

    if fig_type == 'dist':
        fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(10, 10))

        dist_types = ['prev', 'viewed', 'neighbour', 'second_neighbour', 'third_neighbour', 'global']
        dist_titles = ['Previous', 'Interacted', 'Neighbours', 'Second Order Neighbours', 'Third Order Neighbours', 'Global']
        for idx, row in enumerate(axes):
            row[0].hist(dists[f'{dist_types[idx]}_user_content_euclid_dist_{edge_type}'], range=(0, 25), bins=NUM_BINS, density=True)
            row[0].set_title(dist_titles[idx])

        for idx, row in enumerate(axes):
            row[1].hist(dists[f'{dist_types[idx]}_user_content_cosine_sim_{edge_type}'], range=(0, 1), bins=NUM_BINS, density=True)
            row[1].set_title(dist_titles[idx])

        

        for ax in (row[0] for row in axes):
            ax.set_xlim(left=0, right=25)

        for ax in (row[1] for row in axes):
            ax.set_xlim(left=0, right=1)

        axes[0][0].set_ylabel('Density')
        axes[0][0].set_xlabel('Euclidean Distance')

        axes[0][1].set_ylabel('Density')
        axes[0][1].set_xlabel('Cosine Similarity')

    if fig_type == 'movement':
        fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(15, 3))

        dist_types = ['viewed', 'neighbour', 'second_neighbour', 'third_neighbour', 'global']
        dist_titles = ['Interacted', 'Neighbours', 'Second Order Neighbours', 'Third Order Neighbours', 'Global']
        for idx, ax in enumerate(axes):
            ax.hist2d(dists[f'{dist_types[idx]}_user_content_euclid_dist_between_{edge_type}'], dists[f'{dist_types[idx]}_user_content_euclid_dist_away_{edge_type}'], bins=int(np.sqrt(NUM_BINS)), density=True)
            #row[0].hist(dists[f'{dist_types[idx]}_user_content_euclid_dist_between_{edge_type}'], bins=NUM_BINS, density=True)
            ax.set_title(dist_titles[idx])

        # for idx, row in enumerate(axes):
        #     row[1].hist2d(dists[f'{dist_types[idx]}_user_content_cosine_sim_between_{edge_type}'], dists[f'{dist_types[idx]}_user_content_cosine_sim_away_{edge_type}'], bins=int(np.sqrt(NUM_BINS)), density=True)
        #     #row[1].hist(dists[f'{dist_types[idx]}_user_content_cosine_sim_between_{edge_type}'], bins=NUM_BINS, density=True)
        #     row[1].set_title(dist_titles[idx])

        for ax in axes:
            ax.set_xlim(left=0, right=1)

        axes[0].set_ylabel('Away')
        axes[0].set_xlabel('Between')

    plt.tight_layout()

    fig_name = f'{edge_type}_{fig_type}_{change_type}_hists.png'
    fig_path = os.path.join(root_dir_path, 'figs', fig_name)
    fig.savefig(fig_path)


if __name__ == '__main__':
    main()