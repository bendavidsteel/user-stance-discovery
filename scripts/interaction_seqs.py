import datetime
import json
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial
import tqdm

GLOBAL_CONTENT_SAMPLE = 1000
NEIGHBOUR_CONTENT_SAMPLE = 100
NEIGHBOUR_SAMPLE = 1000
USER_SAMPLE = 1000000
MAX_TIME_FRAME = datetime.timedelta(days=7)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    graph_path = os.path.join(data_dir_path, 'graph_data.json')
    with open(graph_path, 'r') as f:
        node_link_data = json.load(f)

    multi_graph = nx.node_link_graph(node_link_data)

    video_embeddings_path = os.path.join(data_dir_path, 'all_video_desc_bertweet_umap_embeddings.npy')
    with open(video_embeddings_path, 'rb') as f:
        video_embeddings = np.load(f)

    video_desc_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    video_desc_df = pd.read_csv(video_desc_path, dtype={'video_id': str, 'author_id': str})
    video_desc_df = video_desc_df.loc[:, ~video_desc_df.columns.str.contains('^Unnamed')]
    video_desc_df['createtime'] = pd.to_datetime(video_desc_df['createtime'])
    video_desc_df = video_desc_df.reset_index().set_index('video_id')
    time_videos_df = video_desc_df.reset_index().set_index(['createtime', 'author_id']).sort_index(level='createtime', ascending=True)
    users_video_df = video_desc_df.groupby('author_id')

    comment_embeddings_path = os.path.join(data_dir_path, 'all_english_comment_bertweet_umap_embeddings.npy')
    with open(comment_embeddings_path, 'rb') as f:
        comment_embeddings = np.load(f)

    comments_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    comments_df = pd.read_csv(comments_path, dtype={'comment_id': str, 'author_id': str})
    comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
    comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
    comments_df = comments_df.reset_index().set_index('comment_id')
    time_comments_df = comments_df.reset_index().set_index(['createtime', 'author_id']).sort_index(level='createtime', ascending=True)
    users_comment_df = comments_df.groupby('author_id')

    # precompute things
    video_comment_product = np.matmul(video_embeddings, np.transpose(comment_embeddings))
    video_norm = np.linalg.norm(video_embeddings, axis=1)
    comment_norm = np.linalg.norm(comment_embeddings, axis=1)
    video_comment_cosine_sim = video_comment_product / np.outer(video_norm, comment_norm)

    user_seqs = {}
    nodes = list(multi_graph.nodes())
    for user_id in tqdm.tqdm(nodes[:USER_SAMPLE]):
        
        inter_items = []
        for u, v, edge_data in multi_graph.out_edges(user_id, data=True):
            viewed_content_user_id = u if u != user_id else v
            inter_items.append((viewed_content_user_id, edge_data))

        # we only care for people with multiple content interactions so we can see diff over time
        if len(inter_items) < 2:
            continue
        # sort interactions by time
        inter_items = sorted(inter_items, key=lambda inter_item: inter_item[1]['unix_createtime'])

        # get first order neighbours
        neighbours = set()
        neighbours.update(multi_graph.successors(user_id))
        neighbours.update(multi_graph.predecessors(user_id))

        # get second order neighbours
        second_order_neighbours = set()
        for neighbour_id in neighbours:
            second_order_neighbours.update(multi_graph.successors(neighbour_id))
            second_order_neighbours.update(multi_graph.predecessors(neighbour_id))

        # make sure they're only second order neighbours
        second_order_neighbours.remove(user_id)
        for neighbour_id in neighbours:
            if neighbour_id in second_order_neighbours:
                second_order_neighbours.remove(neighbour_id)

        # get third order neighbours
        third_order_neighbours = set()
        for neighbour_id in second_order_neighbours:
            third_order_neighbours.update(multi_graph.successors(neighbour_id))
            third_order_neighbours.update(multi_graph.predecessors(neighbour_id))

        # make sure they're only third order neighbours
        if user_id in third_order_neighbours:
            third_order_neighbours.remove(user_id)
        for neighbour_id in neighbours.union(second_order_neighbours):
            if neighbour_id in third_order_neighbours:
                third_order_neighbours.remove(neighbour_id)

        neighbours = tuple(neighbours if len(neighbours) <= NEIGHBOUR_SAMPLE else random.sample(list(neighbours), NEIGHBOUR_SAMPLE))
        second_order_neighbours = tuple(second_order_neighbours if len(second_order_neighbours) <= NEIGHBOUR_SAMPLE else random.sample(list(second_order_neighbours), NEIGHBOUR_SAMPLE))
        third_order_neighbours = tuple(third_order_neighbours if len(third_order_neighbours) <= NEIGHBOUR_SAMPLE else random.sample(list(third_order_neighbours), NEIGHBOUR_SAMPLE))

        user_seq = []
        user_content_indices = []
        user_content_embeds = []
        for viewed_content_user_id, edge_data in inter_items:
            interaction_timestamp = edge_data['unix_createtime']
            interaction_datetime = datetime.datetime.fromtimestamp(interaction_timestamp)
            last_content_timestamp = user_seq[-1]['timestamp'] if len(user_seq) > 0 else 0
            last_content_datetime = datetime.datetime.fromtimestamp(last_content_timestamp)
            frame_start_datetime = interaction_datetime - MAX_TIME_FRAME
            last_datetime = max(last_content_datetime, frame_start_datetime)

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
                'edge_data': edge_data
            }

            if len(user_seq) > 0:
                prev_content_index = user_content_indices[-1]
                prev_content_embed = user_content_embeds[-1]

                if edge_data['type'] == 'video_comment':
                    global_content = time_videos_df.loc[last_datetime:interaction_datetime]

                    if not global_content.empty:
                        global_content_index = global_content['index'].to_numpy()
                        if len(global_content_index) > GLOBAL_CONTENT_SAMPLE:
                            global_content_index = np.random.choice(global_content_index, size=GLOBAL_CONTENT_SAMPLE, replace=False)
                        global_content_embed = video_embeddings[global_content_index, :]

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
                        global_content_index = global_content['index'].to_numpy()
                        if len(global_content_index) > GLOBAL_CONTENT_SAMPLE:
                            global_content_index = np.random.choice(global_content_index, size=GLOBAL_CONTENT_SAMPLE, replace=False)
                        global_content_embed = comment_embeddings[global_content_index, :]

                        global_content_euclid_dists = np.linalg.norm(global_content_embed - user_content_embed, axis=1)
                        global_content_cosine_sims = np.dot(global_content_embed, user_content_embed) / (comment_norm[global_content_index] * comment_norm[user_content_index])

                    viewed_content_index = viewed_content['index']

                    prev_content_euclid_dist = np.linalg.norm(prev_content_embed - user_content_embed)
                    prev_content_cosine_sim = np.dot(prev_content_embed, user_content_embed) / (comment_norm[prev_content_index] * comment_norm[user_content_index])

                    viewed_content_euclid_dist = np.linalg.norm(viewed_content_embed - user_content_embed)
                    viewed_content_cosine_sim = np.dot(viewed_content_embed, user_content_embed) / (comment_norm[viewed_content_index] * comment_norm[user_content_index])

                neighbour_content = global_content[global_content.index.get_level_values('author_id').isin(neighbours)]

                if not neighbour_content.empty:
                    neighbour_index = neighbour_content['index'].to_numpy()
                    if len(neighbour_index) > NEIGHBOUR_CONTENT_SAMPLE:
                        neighbour_index = np.random.choice(neighbour_index, size=NEIGHBOUR_CONTENT_SAMPLE, replace=False)

                second_order_neighbour_content = global_content[global_content.index.get_level_values('author_id').isin(second_order_neighbours)]

                if not second_order_neighbour_content.empty:
                    second_order_neighbour_index = second_order_neighbour_content['index'].to_numpy()
                    if len(second_order_neighbour_index) > NEIGHBOUR_CONTENT_SAMPLE:
                        second_order_neighbour_index = np.random.choice(second_order_neighbour_index, size=NEIGHBOUR_CONTENT_SAMPLE, replace=False)

                third_order_neighbour_content = global_content[global_content.index.get_level_values('author_id').isin(third_order_neighbours)]

                if not third_order_neighbour_content.empty:
                    third_order_neighbour_index = third_order_neighbour_content['index'].to_numpy()
                    if len(third_order_neighbour_index) > NEIGHBOUR_CONTENT_SAMPLE:
                        third_order_neighbour_index = np.random.choice(third_order_neighbour_index, size=NEIGHBOUR_CONTENT_SAMPLE, replace=False)

                interaction_data['prev_content_euclid_dist'] = float(prev_content_euclid_dist)
                interaction_data['prev_content_cosine_sim'] = float(prev_content_cosine_sim)

                interaction_data['viewed_content_euclid_dist'] = float(viewed_content_euclid_dist)
                interaction_data['viewed_content_cosine_sim'] = float(viewed_content_cosine_sim)

                if not global_content.empty:
                    global_content_euclid_dist = np.mean(global_content_euclid_dists)
                    global_content_cosine_sim = np.mean(global_content_cosine_sims)

                    interaction_data['global_content_euclid_dist'] = float(global_content_euclid_dist)
                    interaction_data['global_content_cosine_sim'] = float(global_content_cosine_sim)

                    if not neighbour_content.empty:
                        if edge_data['type'] == 'video_comment':
                            neighbour_content_embed = video_embeddings[neighbour_index, :]

                            neighbour_content_euclid_dists = np.linalg.norm(neighbour_content_embed - user_content_embed, axis=1)
                            neighbour_content_cosine_sims = video_comment_cosine_sim[neighbour_index, user_content_index]
                        elif edge_data['type'] == 'comment_reply':
                            neighbour_content_embed = comment_embeddings[neighbour_index, :]

                            neighbour_content_euclid_dists = np.linalg.norm(neighbour_content_embed - user_content_embed, axis=1)
                            neighbour_content_cosine_sims = np.dot(neighbour_content_embed, user_content_embed) / (comment_norm[neighbour_index] * comment_norm[user_content_index])

                        neighbour_content_euclid_dist = np.mean(neighbour_content_euclid_dists)
                        neighbour_content_cosine_sim = np.mean(neighbour_content_cosine_sims)

                        interaction_data['neighbour_content_euclid_dist'] = float(neighbour_content_euclid_dist)
                        interaction_data['neighbour_content_cosine_sim'] = float(neighbour_content_cosine_sim)

                    if not second_order_neighbour_content.empty:
                        if edge_data['type'] == 'video_comment':
                            second_order_neighbour_content_embed = video_embeddings[second_order_neighbour_index, :]

                            second_order_neighbour_content_euclid_dists = np.linalg.norm(second_order_neighbour_content_embed - user_content_embed, axis=1)
                            second_order_neighbour_content_cosine_sims = video_comment_cosine_sim[second_order_neighbour_index, user_content_index]
                        elif edge_data['type'] == 'comment_reply':
                            second_order_neighbour_content_embed = comment_embeddings[second_order_neighbour_index, :]

                            second_order_neighbour_content_euclid_dists = np.linalg.norm(second_order_neighbour_content_embed - user_content_embed, axis=1)
                            second_order_neighbour_content_cosine_sims = np.dot(second_order_neighbour_content_embed, user_content_embed) / (comment_norm[second_order_neighbour_index] * comment_norm[user_content_index])

                        second_order_neighbour_content_euclid_dist = np.mean(second_order_neighbour_content_euclid_dists)
                        second_order_neighbour_content_cosine_sim = np.mean(second_order_neighbour_content_cosine_sims)

                        interaction_data['second_order_neighbour_content_euclid_dist'] = float(second_order_neighbour_content_euclid_dist)
                        interaction_data['second_order_neighbour_content_cosine_sim'] = float(second_order_neighbour_content_cosine_sim)

                    if not third_order_neighbour_content.empty:
                        if edge_data['type'] == 'video_comment':
                            third_order_neighbour_content_embed = video_embeddings[third_order_neighbour_index, :]

                            third_order_neighbour_content_euclid_dists = np.linalg.norm(third_order_neighbour_content_embed - user_content_embed, axis=1)
                            third_order_neighbour_content_cosine_sims = video_comment_cosine_sim[third_order_neighbour_index, user_content_index]
                        elif edge_data['type'] == 'comment_reply':
                            third_order_neighbour_content_embed = comment_embeddings[third_order_neighbour_index, :]

                            third_order_neighbour_content_euclid_dists = np.linalg.norm(third_order_neighbour_content_embed - user_content_embed, axis=1)
                            third_order_neighbour_content_cosine_sims = np.dot(third_order_neighbour_content_embed, user_content_embed) / (comment_norm[third_order_neighbour_index] * comment_norm[user_content_index])

                        third_order_neighbour_content_euclid_dist = np.mean(third_order_neighbour_content_euclid_dists)
                        third_order_neighbour_content_cosine_sim = np.mean(third_order_neighbour_content_cosine_sims)

                        interaction_data['third_order_neighbour_content_euclid_dist'] = float(third_order_neighbour_content_euclid_dist)
                        interaction_data['third_order_neighbour_content_cosine_sim'] = float(third_order_neighbour_content_cosine_sim)

            user_seq.append(interaction_data)
            user_content_indices.append(user_content_index)
            user_content_embeds.append(user_content_embed)

        user_seqs[user_id] = user_seq

    user_seq_path = os.path.join(data_dir_path, 'user_seqs.json')
    with open(user_seq_path, 'w') as f:
        json.dump(user_seqs, f)

if __name__ == '__main__':
    main()