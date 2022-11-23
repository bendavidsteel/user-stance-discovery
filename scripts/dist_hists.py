import collections
import json
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

EPSILON = 1e-6
NUM_BINS = 100

def movement_metric(prev_dist, target_dist):
    return prev_dist / (prev_dist + target_dist + EPSILON)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    user_seq_path = os.path.join(data_dir_path, 'user_seqs.json')
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    dists = collections.defaultdict(list)

    for user_id, user_seq in tqdm(user_seqs.items()):
        for inter in user_seq:

            # viewed_content_euclid_dist_share = prev_content_euclid_dist / (prev_content_euclid_dist + viewed_content_euclid_dist)
            # viewed_content_cosine_sim_share = prev_content_cosine_sim / (prev_content_cosine_sim + viewed_content_cosine_sim)

            # global_content_euclid_dist_share = prev_content_euclid_dist / (prev_content_euclid_dist + global_content_euclid_dist)
            # global_content_cosine_sim_share = prev_content_cosine_sim / (prev_content_cosine_sim + global_content_cosine_sim)

            # second_order_neighbour_content_euclid_dist_share = prev_content_euclid_dist / (prev_content_euclid_dist + second_order_neighbour_content_euclid_dist)
            # second_order_neighbour_content_cosine_sim_share = prev_content_cosine_sim / (prev_content_cosine_sim + second_order_neighbour_content_cosine_sim)

            inter_type = inter['type']
            for key, value in inter.items():
                if 'euclid' in key or 'cosine' in key:
                    dists[f"{key}_{inter_type}"].append(value)
                    if 'prev' not in key:
                        key_suffix = 19 if 'euclid' in key else 18
                        dists[f"{key}_share_{inter_type}"].append(movement_metric(inter[f"prev_{key[-key_suffix:]}"], value))

    edge_type = 'comment_reply'
    fig_type = 'movement'

    if fig_type == 'dist':
        if edge_type == 'comment_reply':
            fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(20, 5))

            axes[0][0].hist(dists['prev_content_euclid_dist_comment_reply'], bins=NUM_BINS, density=True)
            axes[0][0].set_title('Euclidean Distance Previous')
            axes[1][0].hist(dists['viewed_content_euclid_dist_comment_reply'], bins=NUM_BINS, density=True)
            axes[1][0].set_title('Euclidean Distance Viewed')
            axes[2][0].hist(dists['second_order_neighbour_content_euclid_dist_comment_reply'], bins=NUM_BINS, density=True)
            axes[2][0].set_title('Euclidean Distance Second Order')
            axes[3][0].hist(dists['global_content_euclid_dist_comment_reply'], bins=NUM_BINS, density=True)
            axes[3][0].set_title('Euclidean Distance Global')

            axes[0][1].hist(dists['prev_content_cosine_sim_comment_reply'], bins=NUM_BINS, density=True)
            axes[0][1].set_title('Cosine Similarity Previous')
            axes[1][1].hist(dists['viewed_content_cosine_sim_comment_reply'], bins=NUM_BINS, density=True)
            axes[1][1].set_title('Cosine Similarity Viewed')
            axes[2][1].hist(dists['second_order_neighbour_content_cosine_sim_comment_reply'], bins=NUM_BINS, density=True)
            axes[2][1].set_title('Cosine Similarity Second Order')
            axes[3][1].hist(dists['global_content_cosine_sim_comment_reply'], bins=NUM_BINS, density=True)
            axes[3][1].set_title('Cosine Similarity Global')

            fig_name = 'comment_reply_dist_hists.png'

        elif edge_type == 'video_comment':
            fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(20, 5))

            axes[0][0].hist(dists['prev_content_euclid_dist_video_comment'], bins=NUM_BINS, density=True)
            axes[0][0].set_title('Euclidean Distance Previous')
            axes[1][0].hist(dists['viewed_content_euclid_dist_video_comment'], bins=NUM_BINS, density=True)
            axes[1][0].set_title('Euclidean Distance Viewed')
            axes[2][0].hist(dists['global_content_euclid_dist_video_comment'], bins=NUM_BINS, density=True)
            axes[2][0].set_title('Euclidean Distance Global')
            axes[3][0].hist(dists['second_order_neighbour_content_euclid_dist_video_comment'], bins=NUM_BINS, density=True)
            axes[3][0].set_title('Euclidean Distance Second Order')

            axes[0][1].hist(dists['prev_content_cosine_sim_video_comment'], bins=NUM_BINS, density=True)
            axes[0][1].set_title('Cosine Similarity Previous')
            axes[1][1].hist(dists['viewed_content_cosine_sim_video_comment'], bins=NUM_BINS, density=True)
            axes[1][1].set_title('Cosine Similarity Viewed')
            axes[2][1].hist(dists['global_content_cosine_sim_video_comment'], bins=NUM_BINS, density=True)
            axes[2][1].set_title('Cosine Similarity Global')
            axes[3][1].hist(dists['second_order_neighbour_content_cosine_sim_video_comment'], bins=NUM_BINS, density=True)
            axes[3][1].set_title('Cosine Similarity Second Order')

            fig_name = 'video_comment_dist_hists.png'

        for ax in (row[0] for row in axes):
            ax.set_xlim(left=0, right=25)

        for ax in (row[1] for row in axes):
            ax.set_xlim(left=0, right=1)

        axes[0][0].set_ylabel('Density')
        axes[0][0].set_xlabel('Distance')

        axes[0][1].set_ylabel('Density')
        axes[0][1].set_xlabel('Distance')

    if fig_type == 'movement':
        if edge_type == 'comment_reply':
            fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(20, 5))

            axes[0][0].hist(dists['viewed_content_euclid_dist_share_comment_reply'], bins=NUM_BINS, density=True)
            axes[0][0].set_title('Euclidean Distance Viewed')
            axes[1][0].hist(dists['second_order_neighbour_content_euclid_dist_share_comment_reply'], bins=NUM_BINS, density=True)
            axes[1][0].set_title('Euclidean Distance Second Order')
            axes[2][0].hist(dists['global_content_euclid_dist_share_comment_reply'], bins=NUM_BINS, density=True)
            axes[2][0].set_title('Euclidean Distance Global')

            axes[0][1].hist(dists['viewed_content_cosine_sim_share_comment_reply'], bins=NUM_BINS, density=True)
            axes[0][1].set_title('Cosine Similarity Viewed')
            axes[1][1].hist(dists['second_order_neighbour_content_cosine_sim_share_comment_reply'], bins=NUM_BINS, density=True)
            axes[1][1].set_title('Cosine Similarity Second Order')
            axes[2][1].hist(dists['global_content_cosine_sim_share_comment_reply'], bins=NUM_BINS, density=True)
            axes[2][1].set_title('Cosine Similarity Global')

            fig_name = 'comment_reply_movement_hists.png'

        elif edge_type == 'video_comment':
            fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(20, 5))

            axes[0][0].hist(dists['viewed_content_euclid_dist_share_video_comment'], bins=NUM_BINS, density=True)
            axes[0][0].set_title('Euclidean Distance Viewed')
            axes[1][0].hist(dists['global_content_euclid_dist_share_video_comment'], bins=NUM_BINS, density=True)
            axes[1][0].set_title('Euclidean Distance Global')
            axes[2][0].hist(dists['second_order_neighbour_content_euclid_dist_share_video_comment'], bins=NUM_BINS, density=True)
            axes[2][0].set_title('Euclidean Distance Second Order')

            axes[0][1].hist(dists['viewed_content_cosine_sim_share_video_comment'], bins=NUM_BINS, density=True)
            axes[0][1].set_title('Cosine Similarity Viewed')
            axes[1][1].hist(dists['global_content_cosine_sim_share_video_comment'], bins=NUM_BINS, density=True)
            axes[1][1].set_title('Cosine Similarity Global')
            axes[2][1].hist(dists['second_order_neighbour_content_cosine_sim_share_video_comment'], bins=NUM_BINS, density=True)
            axes[2][1].set_title('Cosine Similarity Second Order')

            fig_name = 'video_comment_movement_hists.png'

        for ax in (row[0] for row in axes):
            ax.set_xlim(left=0, right=1)

        for ax in (row[1] for row in axes):
            ax.set_xlim(left=0, right=1)

        axes[0][0].set_ylabel('Density')
        axes[0][0].set_xlabel('Movement')

        axes[0][1].set_ylabel('Density')
        axes[0][1].set_xlabel('Movement')

    plt.tight_layout()

    fig_path = os.path.join(root_dir_path, 'figs', fig_name)
    fig.savefig(fig_path)


if __name__ == '__main__':
    main()