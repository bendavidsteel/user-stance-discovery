import collections
import datetime
import json
import os

import adjustText
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import dist_hists

EPSILON = 1e-6
MAX_EXP = 100
NUM_BINS = 100

def get_dist_examples(dist_interval, user_seqs, dist_name, edge_type):
    for user_id, user_seq in user_seqs.items():
        for inter in user_seq:
            if edge_type != inter['edge_data']['type']:
                continue
            if dist_name not in inter:
                continue
            rel_dist = inter[dist_name]
            if rel_dist >= dist_interval[0] and rel_dist < dist_interval[1]:
                yield user_id, inter
    else:
        raise ValueError()

def get_movement_examples(between_interval, away_interval, user_seqs, dist_name, edge_type):
    for user_id, user_seq in user_seqs.items():
        for inter in user_seq:
            if edge_type != inter['edge_data']['type']:
                continue
            if dist_name not in inter:
                continue
            other_user_dist = inter[dist_name]
            prev_other_dist = inter[dist_name.replace('user', 'prev')]
            user_prev_dist = inter[f"prev_user_{dist_name[-19:]}"]
            between_metric = dist_hists.between_metric(other_user_dist, prev_other_dist, user_prev_dist)
            away_metric = dist_hists.away_metric(other_user_dist, prev_other_dist, user_prev_dist)
            if between_metric >= between_interval[0] and between_metric < between_interval[1] and away_metric >= away_interval[0] and away_metric < away_interval[1]:
                yield user_id, inter
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

def try_get_text(user_id, inter, users_comment_df, users_video_df):
    user_content_text, prev_content_text, viewed_content_text = get_text(user_id, inter, users_comment_df, users_video_df)

    min_length = 10
    if (len(user_content_text) < min_length) and (len(prev_content_text) < min_length) and (len(viewed_content_text) < min_length):
        return ()

    if user_content_text == prev_content_text:
        return ()

    return user_content_text, prev_content_text, viewed_content_text

def _plot_movement(ax, inter, user_content_text, prev_content_text, viewed_content_text):
    max_length = 30
    user_content_text = user_content_text if len(user_content_text) < max_length else user_content_text[:max_length] + '...'
    prev_content_text = prev_content_text if len(prev_content_text) < max_length else prev_content_text[:max_length] + '...'
    viewed_content_text = viewed_content_text if len(viewed_content_text) < max_length else viewed_content_text[:max_length] + '...'

    ax.arrow(inter['prev_content_embed'][0], inter['prev_content_embed'][1], inter['user_content_embed'][0] - inter['prev_content_embed'][0], inter['user_content_embed'][1] - inter['prev_content_embed'][1])
    ax.plot([inter['user_content_embed'][0], inter['viewed_content_embed'][0]], [inter['user_content_embed'][1], inter['viewed_content_embed'][1]], '--')

    ax.plot(*inter['user_content_embed'], 'bo', label='User Content')
    ax.plot(*inter['prev_content_embed'], 'go', label='Prev Content')
    ax.plot(*inter['viewed_content_embed'], 'ro', label='Interacted Content')

    if 'neighbour_content_embed' in inter:
        ax.plot(*inter['neighbour_content_embed'], 'co', label='Neighbour Content')

    if 'second_neighbour_content_embed' in inter:
        ax.plot(*inter['second_neighbour_content_embed'], 'mo', label='Second Order Neighbour Content')

    if 'third_neighbour_content_embed' in inter:
        ax.plot(*inter['third_neighbour_content_embed'], 'yo', label='Third Order Neighbour Content')

    if 'global_content_embed' in inter:
        ax.plot(*inter['global_content_embed'], 'ko', label='Global Content Embed')

    return [(inter['user_content_embed'], user_content_text), (inter['prev_content_embed'], prev_content_text), (inter['viewed_content_embed'], viewed_content_text)]


def plot_movement(ax, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df):
    for user_id, inter in get_dist_examples(dist_interval, user_seqs, dist_name, edge_type):
        # min dist
        good = _try_plot_movement(ax, user_id, inter, users_comment_df, users_video_df)
        if good:
            return

def plot_dist_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, n, bins):

    first_idx = np.argmax(n != 0)
    last_idx = (n.shape[0]-1) - np.argmax(n[::-1] != 0)

    num_cols = 1
    # peak
    peak_idx = np.argmax(n)
    if (peak_idx != first_idx) and (peak_idx != last_idx):
        num_rows = 3
        dist_interval = (bins[peak_idx], bins[peak_idx+1])
        ax = fig.add_subplot(num_rows, num_cols, num_cols+1)
        plot_movement(ax, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df)
        ax.set_title('Mode')
    else:
        num_rows = 2

    dist_interval = (bins[first_idx], bins[first_idx+1])
    ax = fig.add_subplot(num_rows, num_cols, 1)
    plot_movement(ax, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df)
    ax.legend(bbox_to_anchor=(-0.2, 1))
    ax.set_title('Least')

    # max dist
    dist_interval = (bins[last_idx-10], bins[last_idx+1])
    ax = fig.add_subplot(num_rows, num_cols, (num_rows * num_cols) - num_cols + 1)
    plot_movement(ax, dist_interval, user_seqs, dist_name, edge_type, users_comment_df, users_video_df)
    ax.set_title('Most')

def plot_movement_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, n, xbins, ybins):

    top_n = 3
    num_examples = 3
    top_n_idx = [
        [0, 0],
        [9, 0],
        [5, 9]
    ]
    names = [
        'Stayed near Previous',
        'Moved to Interacted',
        'Moved Away'
    ]
    
    for row_idx, top_idx in enumerate(top_n_idx):
        between_interval = (xbins[top_idx[0]], xbins[top_idx[0]+1])
        away_interval = (ybins[top_idx[1]], ybins[top_idx[1]+1])
        examples_so_far = 0
        for user_id, inter in get_movement_examples(between_interval, away_interval, user_seqs, dist_name, edge_type):
            texts = try_get_text(user_id, inter, users_comment_df, users_video_df)
            if not texts:
                continue
            ax_num = examples_so_far + 1 + (row_idx * num_examples)
            ax = fig.add_subplot(top_n, num_examples, ax_num)
            # min dist
            text_pos = _plot_movement(ax, inter, *texts)
            if ax_num == 1:
                ax.legend(bbox_to_anchor=(-0.2, 1))
            if ax_num % num_examples == 2:
                ax.set_title(names[row_idx])
            text_plots = []
            for pos, text in text_pos:
                text_plots.append(ax.text(*pos, text))
            adjustText.adjust_text(text_plots)
            examples_so_far += 1
            if examples_so_far == num_examples:
                break
        

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    user_seq_path = os.path.join(data_dir_path, 'user_seqs_instantaneous_2.json')
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    dists = collections.defaultdict(list)

    edge_type = 'comment_reply'
    fig_type = 'movement'

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
                        dists[(key, 'between', inter_type)].append(dist_hists.between_metric(other_user_dist, prev_other_dist, user_prev_dist))
                        dists[(key, 'away', inter_type)].append(dist_hists.away_metric(other_user_dist, prev_other_dist, user_prev_dist))

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
            fig = plt.figure(figsize=(15, 10))
            dist_name = f'{dist_types[idx]}_user_content_euclid_dist'
            dist_key = (dist_name, edge_type)
            n, bins = np.histogram(dists[dist_key], range=(0, 25), bins=NUM_BINS, density=True)

            # if dist_types[idx] in ['prev', 'viewed']:
            plot_dist_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, n, bins)

            plt.tight_layout()

            fig_name = f'{edge_type}_{dist_types[idx]}_euclid_dist_examples.png'
            fig_path = os.path.join(root_dir_path, 'figs', fig_name)
            fig.savefig(fig_path)

        for idx in range(len(dist_types)):
            fig = plt.figure(figsize=(15, 10))
            dist_name = f'{dist_types[idx]}_user_content_cosine_sim'
            dist_key = (dist_name, edge_type)
            n, bins = np.histogram(dists[dist_key], range=(0, 1), bins=NUM_BINS, density=True)

            # if dist_types[idx] in ['prev', 'viewed']:
            plot_dist_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, n, bins)

            plt.tight_layout()

            fig_name = f'{edge_type}_{dist_types[idx]}_cosine_sim_examples.png'
            fig_path = os.path.join(root_dir_path, 'figs', fig_name)
            fig.savefig(fig_path)

    if fig_type == 'movement':
        fig = plt.figure(figsize=(13, 10))

        dist_type = 'viewed'
        dist_title = 'Interacted'
        dist_name = f'{dist_type}_user_content_euclid_dist'

        h, xedges, yedges = np.histogram2d(dists[(f'{dist_type}_user_content_euclid_dist', 'between', edge_type)], dists[(f'{dist_type}_user_content_euclid_dist', 'away', edge_type)], bins=int(np.sqrt(NUM_BINS)), density=True)

        plot_movement_examples(fig, dist_name, edge_type, users_comment_df, users_video_df, user_seqs, h, xedges, yedges)

        plt.tight_layout()

        fig_name = f'{edge_type}_movement_examples.png'
        fig_path = os.path.join(root_dir_path, 'figs', fig_name)
        fig.savefig(fig_path)


if __name__ == '__main__':
    main()