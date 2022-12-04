import collections
import datetime
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

EPSILON = 1e-6
MAX_EXP = 100
NUM_BINS = 100

def plot_text_pair(ax, text_a, text_b, pos, height):
    ax.axvline(x=pos)
    ax.text(pos - 0.1, height, text_a)
    ax.text(pos + 0.1, height, text_b)

def plot_text_triplet(ax, text_a, text_b, text_c, pos, height):
    ax.axvline(x=pos)
    ax.text(pos - 0.1, height, text_a)
    ax.text(pos, height, text_b)
    ax.text(pos + 0.1, height, text_c)

def get_dist_example(dist_interval, user_seqs, dist_name, edge_type):
    for user_id, user_seq in user_seqs.items():
        for inter in user_seq:
            if edge_type != inter['edge_data']['type']:
                continue
            if dist_name not in inter:
                continue
            rel_dist = inter[dist_name]
            if rel_dist >= dist_interval[0] and rel_dist < dist_interval[1]:
                return user_id, inter
    else:
        raise ValueError()

def get_text(user_id, inter, users_comment_df, users_video_df):
    user_comments_df = users_comment_df.get_group(user_id)
    user_comments_df = user_comments_df.sort_values('createtime')
    interaction_datetime = datetime.datetime.fromtimestamp(inter['timestamp']) + datetime.timedelta(hours=5)
    recent_comments_df = user_comments_df[user_comments_df['createtime'] <= interaction_datetime]

    if inter['edge_data']['type'] == 'comment_reply':
        user_content_text = user_comments_df.loc[inter['edge_data']['comment_id_reply'], 'text']

        viewed_user_content = users_comment_df.get_group(inter['viewed_content_user_id'])
        viewed_content_text = viewed_user_content.loc[inter['edge_data']['comment_id'], 'text']

        prev_content_text = recent_comments_df.iloc[-2]['text']

    elif inter['edge_data']['type'] == 'video_comment':
        user_content_text = user_comments_df.loc[inter['edge_data']['comment_id'], 'text']

        viewed_user_content = users_video_df.get_group(inter['viewed_content_user_id'])
        viewed_content_text = viewed_user_content.loc[inter['edge_data']['video_id'], 'desc']

        prev_content_text = recent_comments_df.iloc[-2]['text']

    return user_content_text, prev_content_text, viewed_content_text

def _plot_movement(ax, user_id, inter, users_comment_df, users_video_df):
    user_content_text, prev_content_text, viewed_content_text = get_text(user_id, inter, users_comment_df, users_video_df)
    ax.plot(*inter['user_content_embed'], 'go', label='User Content')
    ax.text(*inter['user_content_embed'], user_content_text)

    ax.plot(*inter['prev_content_embed'], 'ro', label='Prev Content')
    ax.text(*inter['prev_content_embed'], prev_content_text)

    ax.plot(*inter['viewed_content_embed'], 'yo', label='Interacted Content')
    ax.text(*inter['viewed_content_embed'], viewed_content_text)

    ax.plot(*inter['neighbour_user_content_embed'], 'o', label='Neighbour Content')
    ax.plot(*inter['second_neighbour_user_content_embed'], 'bo', label='Second Order Neighbour Content')
    ax.plot(*inter['third_order_neighbour_user_content_embed'], 'po', label='Third Order Neighbour Content')
    ax.plot(*inter['global_content_embed'], 'go', label='Global Content Embed')


def plot_movement(fig, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df):
    user_id, inter = get_dist_example(dist_interval, user_seqs, dist_name, edge_type)
    ax = fig.add_subplot()
    # min dist
    _plot_movement(ax, user_id, inter, users_comment_df, users_video_df)

def plot_dist_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, n, bins):

    dist_interval = (bins[0], bins[1])
    plot_movement(fig, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df)

    # peaks
    for idx in range(1, len(n) - 2):
        if n[idx] > n[idx-1] and n[idx] > n[idx+1]:
            dist_interval = (bins[idx], bins[idx+1])
            plot_movement(fig, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df)

    # max dist
    dist_interval = (bins[-2], bins[-1])
    plot_movement(fig, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df)

def plot_movement_examples(ax, user_seqs, dists, n, bins, patches):
    # min dist
    plot_movement(ax, )

    # peaks
    for _ in bins:
        plot_movement(ax, )

    # max dist
    plot_movement(ax, )

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
        assert other_user_dist == user_prev_dist
        dist_from_line = other_user_dist
    dist_from_line = min(dist_from_line, MAX_EXP)
    return np.e ** (dist_from_line) / (np.e ** (dist_from_line) + 1)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    user_seq_path = os.path.join(data_dir_path, 'user_seqs_instantaneous_2.json')
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    dists = collections.defaultdict(list)

    edge_type = 'video_comment'
    fig_type = 'dist'

    for user_id, user_seq in tqdm(user_seqs.items()):
        for inter in user_seq:

            inter_type = inter['edge_data']['type']
            for key, value in inter.items():
                if 'euclid' in key or 'cosine' in key:
                    dists[(key, inter_type)].append(value)
                    if fig_type == 'movement' and '_user_' in key and 'prev' not in key:
                        key_suffix = 19 if 'euclid' in key else 18
                        other_user_dist = value
                        prev_other_dist = inter[key.replace('user', 'prev')]
                        user_prev_dist = inter[f"prev_user_{key[-key_suffix:]}"]
                        dists[(key, 'between', inter_type)].append(between_metric(other_user_dist, prev_other_dist, user_prev_dist))
                        dists[(key, 'away', inter_type)].append(away_metric(other_user_dist, prev_other_dist, user_prev_dist))

    video_desc_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    video_desc_df = pd.read_csv(video_desc_path, dtype={'video_id': str, 'author_id': str})
    video_desc_df = video_desc_df.loc[:, ~video_desc_df.columns.str.contains('^Unnamed')]
    video_desc_df['createtime'] = pd.to_datetime(video_desc_df['createtime'])
    video_desc_df = video_desc_df.reset_index().set_index('video_id')
    users_video_df = video_desc_df.groupby('author_id')

    comments_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    comments_df = pd.read_csv(comments_path, dtype={'comment_id': str, 'author_id': str, 'reply_comment_id': str})
    comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
    comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
    comments_df = comments_df.reset_index().set_index('comment_id')
    users_comment_df = comments_df.groupby('author_id')

    if fig_type == 'dist':
        dist_types = ['prev', 'viewed', 'neighbour', 'second_neighbour', 'third_neighbour', 'global']
        dist_titles = ['Previous', 'Interacted', 'Neighbours', 'Second Order Neighbours', 'Third Order Neighbours', 'Global']
        for idx in range(len(dist_types)):
            fig = plt.figure()
            ax = fig.add_subplot()
            dist_name = f'{dist_types[idx]}_user_content_euclid_dist'
            dist_key = (dist_name, edge_type)
            n, bins, patches = ax.hist(dists[dist_key], range=(0, 25), bins=NUM_BINS, density=True)
            ax.set_title(dist_titles[idx])
            ax.set_xlim(left=0, right=25)
            ax.set_ylabel('Density')
            ax.set_xlabel('Euclidean Distance')

            if dist_types[idx] in ['prev', 'viewed']:
                plot_dist_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, n, bins)

            plt.tight_layout()

            fig_name = f'{edge_type}_{dist_types[idx]}_dist_examples.png'
            fig_path = os.path.join(root_dir_path, 'figs', fig_name)
            fig.savefig(fig_path)

        for idx in range(len(dist_types)):
            fig = plt.figure()
            ax = fig.add_subplot()
            dist_name = f'{dist_types[idx]}_user_content_cosine_sim'
            dist_key = (dist_name, edge_type)
            n, bins, patches = ax.hist(dists[dist_key], range=(0, 1), bins=NUM_BINS, density=True)
            ax.set_title(dist_titles[idx])
            ax.set_xlim(left=0, right=1)
            ax.set_ylabel('Density')
            ax.set_xlabel('Cosine Similarity')

            if dist_types[idx] in ['prev', 'viewed']:
                plot_dist_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, n, bins, patches)

    if fig_type == 'movement':
        fig, axes = plt.subplots(ncols=1, nrows=5, figsize=(5, 15))

        dist_types = ['viewed', 'neighbour', 'second_neighbour', 'third_neighbour', 'global']
        dist_titles = ['Interacted', 'Neighbours', 'Second Order Neighbours', 'Third Order Neighbours', 'Global']
        for idx, ax in enumerate(axes):
            h, xedges, yedges, image = ax.hist2d(dists[f'{dist_types[idx]}_user_content_euclid_dist_between_{edge_type}'], dists[f'{dist_types[idx]}_user_content_euclid_dist_away_{edge_type}'], bins=int(np.sqrt(NUM_BINS)), density=True)
            ax.set_title(dist_titles[idx])

            if dist_types[idx] == 'viewed':
                plot_movement_examples(ax, dists, n, bins, patches)

        fig_name = f'{edge_type}_movement_hists.png'

        for ax in (row for row in axes):
            ax.set_xlim(left=0, right=1)

        axes[0].set_ylabel('Away')
        axes[0].set_xlabel('Between')

    


if __name__ == '__main__':
    main()