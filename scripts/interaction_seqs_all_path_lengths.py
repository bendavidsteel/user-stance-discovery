import collections
import datetime
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial
import tqdm

def get_nth_order_neighbour_dists(global_content, global_content_index, global_content_euclid_dists, global_content_cosine_sims, nth_order_neighbours, nth_name, interaction_data):
    nth_order_neighbour_content = global_content.loc[(slice(None), nth_order_neighbours), :]

    if not nth_order_neighbour_content.empty:
        nth_order_neighbour_index = nth_order_neighbour_content['index'].to_numpy()
        nth_order_index_in_global = np.where(np.in1d(global_content_index, nth_order_neighbour_index))[0]

        nth_order_neighbour_content_euclid_dists = global_content_euclid_dists[nth_order_index_in_global]
        nth_order_neighbour_content_cosine_sims = global_content_cosine_sims[nth_order_index_in_global]

        nth_order_neighbour_content_euclid_dist = np.mean(nth_order_neighbour_content_euclid_dists)
        nth_order_neighbour_content_cosine_sim = np.mean(nth_order_neighbour_content_cosine_sims)

        interaction_data[f'{nth_name}_order_neighbour_content_euclid_dist'] = float(nth_order_neighbour_content_euclid_dist)
        interaction_data[f'{nth_name}_order_neighbour_content_cosine_sim'] = float(nth_order_neighbour_content_cosine_sim)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    graph_path = os.path.join(data_dir_path, 'graph_data.json')
    with open(graph_path, 'r') as f:
        node_link_data = json.load(f)

    multi_graph = nx.node_link_graph(node_link_data)

    all_pairs_path_lengths = nx.all_pairs_shortest_path_length(multi_graph.to_undirected(), cutoff=5)
    all_neighbours = {}
    for source, targets in all_pairs_path_lengths:
        all_neighbours[source] = collections.defaultdict(set)
        for target, shortest_path_length in targets.items():
            all_neighbours[source][shortest_path_length].add(target)
        for path_length, neighbours in all_neighbours[source].items():
            all_neighbours[source][path_length] = tuple(all_neighbours[source][path_length])

    video_embeddings_path = os.path.join(data_dir_path, 'all_video_desc_twitter_roberta_umap_embeddings.npy')
    with open(video_embeddings_path, 'rb') as f:
        video_embeddings = np.load(f)

    video_desc_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    video_desc_df = pd.read_csv(video_desc_path, dtype={'video_id': str, 'author_id': str})
    video_desc_df = video_desc_df.loc[:, ~video_desc_df.columns.str.contains('^Unnamed')]
    video_desc_df['createtime'] = pd.to_datetime(video_desc_df['createtime'])
    video_desc_df = video_desc_df.reset_index().set_index('video_id')
    video_desc_df = video_desc_df[~video_desc_df.index.duplicated(keep='first')]
    time_videos_df = video_desc_df.reset_index().set_index(['createtime', 'author_id']).sort_index(level='createtime', ascending=True)
    users_video_df = video_desc_df.groupby('author_id')

    comment_embeddings_path = os.path.join(data_dir_path, 'all_english_comment_twitter_roberta_umap_embeddings.npy')
    with open(comment_embeddings_path, 'rb') as f:
        comment_embeddings = np.load(f)

    comments_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    comments_df = pd.read_csv(comments_path, dtype={'comment_id': str, 'author_id': str})
    comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
    comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
    comments_df = comments_df.reset_index().set_index('comment_id')
    comments_df = comments_df[~comments_df.index.duplicated(keep='first')]
    time_comments_df = comments_df.reset_index().set_index(['createtime', 'author_id']).sort_index(level='createtime', ascending=True)
    users_comment_df = comments_df.groupby('author_id')

    # precompute things
    video_comment_product = np.matmul(video_embeddings, np.transpose(comment_embeddings))
    video_norm = np.linalg.norm(video_embeddings, axis=1)
    comment_norm = np.linalg.norm(comment_embeddings, axis=1)
    video_comment_cosine_sim = video_comment_product / np.outer(video_norm, comment_norm)

    user_seqs = {}
    for user_id in tqdm.tqdm(multi_graph.nodes()):
        
        inter_items = []
        for u, v, edge_data in multi_graph.out_edges(user_id, data=True):
            viewed_content_user_id = u if u != user_id else v
            inter_items.append((viewed_content_user_id, edge_data))

        # we only care for people with multiple content interactions so we can see diff over time
        if len(inter_items) < 2:
            continue
        # sort interactions by time
        inter_items = sorted(inter_items, key=lambda inter_item: inter_item[1]['unix_createtime'])

        these_neighbours = all_neighbours[user_id]

        user_seq = []
        user_content_indices = []
        user_content_embeds = []
        for viewed_content_user_id, edge_data in inter_items:
            interaction_timestamp = edge_data['unix_createtime']
            interaction_datetime = datetime.datetime.fromtimestamp(interaction_timestamp)
            last_timestamp = user_seq[-1]['timestamp'] if len(user_seq) > 0 else 0
            last_datetime = datetime.datetime.fromtimestamp(last_timestamp)

            if edge_data['type'] == 'video_comment':
                try:
                    user_videos_df = users_video_df.get_group(viewed_content_user_id)
                    user_comments_df = users_comment_df.get_group(user_id)
                except KeyError:
                    continue

                try:
                    viewed_content = user_videos_df.loc[edge_data['video_id']]
                    user_content = user_comments_df.loc[edge_data['comment_id']]
                except KeyError:
                    continue

                viewed_content_embed = video_embeddings[viewed_content['index'], :]
                user_content_embed = comment_embeddings[user_content['index'], :]
            elif edge_data['type'] == 'comment_reply':
                try:
                    viewed_user_comments_df = users_comment_df.get_group(viewed_content_user_id)
                    user_comments_df = users_comment_df.get_group(user_id)
                except KeyError:
                    continue

                try:
                    viewed_content = viewed_user_comments_df.loc[edge_data['comment_id']]
                    user_content = user_comments_df.loc[edge_data['comment_id_reply']]
                except KeyError:
                    continue

                viewed_content_embed = comment_embeddings[viewed_content['index'], :]
                user_content_embed = comment_embeddings[user_content['index'], :]
            else:
                continue

            user_content_index = user_content['index']

            interaction_data = {
                'viewed_content_user_id': str(viewed_content_user_id),
                'timestamp': int(interaction_timestamp),
                'type': edge_data['type']
            }

            if len(user_seq) > 0:
                prev_content_index = user_content_indices[-1]
                prev_content_embed = user_content_embeds[-1]

                if edge_data['type'] == 'video_comment':
                    global_content = time_videos_df.loc[last_datetime:interaction_datetime]

                    if not global_content.empty:
                        global_content_index = global_content['index'].to_numpy()
                        global_content_embed = video_embeddings[global_content['index'].to_numpy(), :]

                        global_content_euclid_dists = np.linalg.norm(global_content_embed - user_content_embed, axis=1)
                        global_content_cosine_sims = video_comment_cosine_sim[global_content_index, user_content_index]

                    viewed_content_index = viewed_content['index']

                    prev_content_euclid_dist = np.linalg.norm(prev_content_embed - user_content_embed)
                    prev_content_cosine_sim = np.dot(prev_content_embed, user_content_embed) / (comment_norm[prev_content_index] * comment_norm[user_content_index])

                    viewed_content_euclid_dist = np.linalg.norm(viewed_content_embed - user_content_embed)
                    viewed_content_cosine_sim = video_comment_cosine_sim[viewed_content_index, user_content_index]

                elif edge_data['type'] == 'comment_reply':
                    global_content = time_comments_df.loc[last_datetime:interaction_datetime]

                    if not global_content.empty:
                        global_content_embed = comment_embeddings[global_content['index'].to_numpy(), :]
                        global_content_index = global_content['index'].to_numpy()

                        global_content_euclid_dists = np.linalg.norm(global_content_embed - user_content_embed, axis=1)
                        global_content_cosine_sims = np.dot(global_content_embed, user_content_embed) / (comment_norm[global_content_index] * comment_norm[user_content_index])

                    viewed_content_index = viewed_content['index']

                    prev_content_euclid_dist = np.linalg.norm(prev_content_embed - user_content_embed)
                    prev_content_cosine_sim = np.dot(prev_content_embed, user_content_embed) / (comment_norm[prev_content_index] * comment_norm[user_content_index])

                    viewed_content_euclid_dist = np.linalg.norm(viewed_content_embed - user_content_embed)
                    viewed_content_cosine_sim = np.dot(viewed_content_embed, user_content_embed) / (comment_norm[viewed_content_index] * comment_norm[user_content_index])

                interaction_data['prev_content_euclid_dist'] = float(prev_content_euclid_dist)
                interaction_data['prev_content_cosine_sim'] = float(prev_content_cosine_sim)

                interaction_data['viewed_content_euclid_dist'] = float(viewed_content_euclid_dist)
                interaction_data['viewed_content_cosine_sim'] = float(viewed_content_cosine_sim)

                if not global_content.empty:
                    global_content_euclid_dist = np.mean(global_content_euclid_dists)
                    global_content_cosine_sim = np.mean(global_content_cosine_sims)

                    interaction_data['global_content_euclid_dist'] = float(global_content_euclid_dist)
                    interaction_data['global_content_cosine_sim'] = float(global_content_cosine_sim)

                    orders = ['first', 'second', 'third']
                    for idx, order in enumerate(orders):
                        nth = idx + 1
                        get_nth_order_neighbour_dists(global_content, global_content_index, global_content_euclid_dists, global_content_cosine_sims, these_neighbours[nth], order, interaction_data)

            user_seq.append(interaction_data)
            user_content_indices.append(user_content_index)
            user_content_embeds.append(user_content_embed)

        user_seqs[user_id] = user_seq

    user_seq_path = os.path.join(data_dir_path, 'user_seqs.json')
    with open(user_seq_path, 'w') as f:
        json.dump(user_seqs, f)

if __name__ == '__main__':
    main()