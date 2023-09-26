import datetime
import json
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial
import tqdm

import warnings
warnings.filterwarnings("error")

GLOBAL_CONTENT_SAMPLE = 1000
NEIGHBOUR_CONTENT_SAMPLE = 1000
NEIGHBOUR_SAMPLE = 10000
USER_SAMPLE = 2000000
MAX_TIME_FRAME = datetime.timedelta(days=7)
INSTANTANEOUS = False
NUM_DIMS = 5

def get_neighbours(multi_graph, user_id):

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

    return neighbours, second_order_neighbours, third_order_neighbours

def get_neighbour_dists(interaction_data, global_content, neighbours, neighbour_prefix, edge_data, video_embeddings, comment_embeddings, user_content_embed, user_content_index, prev_content_embed, prev_content_index, comment_norm):
    neighbour_content = global_content[global_content.index.get_level_values('author_id').isin(neighbours)]

    if not neighbour_content.empty:
        neighbour_index = neighbour_content['index'].to_numpy()
        if len(neighbour_index) > NEIGHBOUR_CONTENT_SAMPLE:
            neighbour_index = np.random.choice(neighbour_index, size=NEIGHBOUR_CONTENT_SAMPLE, replace=False)

        if edge_data['type'] == 'video_comment':
            neighbour_content_embed = np.mean(video_embeddings[neighbour_index, :], axis=0)
        elif edge_data['type'] == 'comment_reply':
            neighbour_content_embed = np.mean(comment_embeddings[neighbour_index, :], axis=0)

        neighbour_content_euclid_dist = np.linalg.norm(neighbour_content_embed - user_content_embed)
        neighbour_prev_content_euclid_dist = np.linalg.norm(neighbour_content_embed - prev_content_embed)

        norm_neighbour_content_embed = np.linalg.norm(neighbour_content_embed)
        if INSTANTANEOUS:
            norm_user_content_embed = comment_norm[user_content_index]
            norm_prev_content_embed = comment_norm[prev_content_index]
        else:
            norm_user_content_embed = np.linalg.norm(user_content_embed)
            norm_prev_content_embed = np.linalg.norm(prev_content_embed)

        neighbour_content_cosine_sim = np.dot(neighbour_content_embed, user_content_embed) / (norm_neighbour_content_embed * norm_user_content_embed)
        neighbour_prev_content_cosine_sim = np.dot(neighbour_content_embed, prev_content_embed) / (norm_neighbour_content_embed * norm_prev_content_embed)

        interaction_data[f'{neighbour_prefix}neighbour_content_embed'] = [float(round(e, 8)) for e in neighbour_content_embed]

        interaction_data[f'{neighbour_prefix}neighbour_user_content_euclid_dist'] = float(round(neighbour_content_euclid_dist, 4))
        interaction_data[f'{neighbour_prefix}neighbour_user_content_cosine_sim'] = float(round(neighbour_content_cosine_sim, 4))

        interaction_data[f'{neighbour_prefix}neighbour_prev_content_euclid_dist'] = float(round(neighbour_prev_content_euclid_dist, 4))
        interaction_data[f'{neighbour_prefix}neighbour_prev_content_cosine_sim'] = float(round(neighbour_prev_content_cosine_sim, 4))

def precompute_mats(video_embeddings, comment_embeddings):
    video_comment_product = np.matmul(video_embeddings, np.transpose(comment_embeddings))
    video_norm = np.linalg.norm(video_embeddings, axis=1)
    comment_norm = np.linalg.norm(comment_embeddings, axis=1)
    video_comment_cosine_sim = video_comment_product / np.outer(video_norm, comment_norm)
    return video_norm, comment_norm, video_comment_cosine_sim


def process_global_data(time_content_df, content_embeddings, last_datetime, interaction_datetime, user_content_embed, prev_content_embed, norm_user_content_embed, norm_prev_content_embed, interaction_data):
    global_content = time_content_df.xs(slice(last_datetime, interaction_datetime), level='createtime').iloc[1:-1]

    if not global_content.empty:
        global_content_index = global_content['index'].to_numpy()
        if len(global_content_index) > GLOBAL_CONTENT_SAMPLE:
            global_content_index = np.random.choice(global_content_index, size=GLOBAL_CONTENT_SAMPLE, replace=False)
        global_content_embed = np.mean(content_embeddings[global_content_index, :], axis=0)

        global_content_euclid_dist = np.linalg.norm(global_content_embed - user_content_embed)
        global_prev_content_euclid_dist = np.linalg.norm(global_content_embed - prev_content_embed)

        norm_global_content_embed = np.linalg.norm(global_content_embed)

        global_content_cosine_sim = np.dot(global_content_embed, user_content_embed) / (norm_global_content_embed * norm_user_content_embed)
        global_prev_content_cosine_sim = np.dot(global_content_embed, prev_content_embed) / (norm_global_content_embed * norm_prev_content_embed)

        interaction_data['global_content_embed'] = [float(round(e, 8)) for e in global_content_embed]

        interaction_data['global_user_content_euclid_dist'] = float(round(global_content_euclid_dist, 4))
        interaction_data['global_user_content_cosine_sim'] = float(round(global_content_cosine_sim, 4))

        interaction_data['global_prev_content_euclid_dist'] = float(round(global_prev_content_euclid_dist, 4))
        interaction_data['global_prev_content_cosine_sim'] = float(round(global_prev_content_cosine_sim, 4))

    return interaction_data

def assert_and_get_viewed_id(user_id, user_content, edge_data, time_content_df, viewed_content_user_id, user_id_name, viewed_id_name):
    if INSTANTANEOUS:
        assert user_content.name[1] == edge_data[user_id_name], f"Failed to get content match for user ID: {user_id}"
    else:
        assert user_content.iloc[0].name == edge_data[user_id_name], f"Failed to get content match for user ID: {user_id}"
    viewed_content = time_content_df.xs(viewed_content_user_id, level='author_id', drop_level=False).xs(edge_data[viewed_id_name], level=viewed_id_name, drop_level=False)
    if len(viewed_content) != 1:
        raise ValueError()
    viewed_content = viewed_content.iloc[0]
    assert viewed_content.name[2] == edge_data[viewed_id_name], f"Failed to get content match for user ID: {user_id}"
    viewed_content_index = viewed_content['index']
    return viewed_content_index

def process_interaction(user_id, edge_data, user_comments_df, time_comments_df, time_videos_df, comment_embeddings, video_embeddings, first_content_datetime, last_content_datetime, viewed_content_user_id, video_norm, comment_norm, video_comment_cosine_sim, neighbours, second_order_neighbours, third_order_neighbours):
    interaction_timestamp = edge_data['unix_createtime']
    # very annoyed that i have to do this but i messed up times a while back and now have to correct
    interaction_datetime = datetime.datetime.fromtimestamp(interaction_timestamp) + datetime.timedelta(hours=5)

    if edge_data['type'] == 'video_comment':
        user_content_id = edge_data['comment_id']
    elif edge_data['type'] == 'comment_reply':
        user_content_id = edge_data['comment_id_reply']

    if INSTANTANEOUS:
        user_content = user_comments_df.xs(user_content_id, level='comment_id', drop_level=False)
        if len(user_content) != 1:
            raise KeyError()
        user_content = user_content.iloc[0]
    else:
        user_content = user_comments_df.xs(slice(interaction_datetime, last_content_datetime), level='createtime')
        if user_content_id not in user_content.index:
            raise KeyError()

    if INSTANTANEOUS:
        user_content_index = int(user_content['index'])
        user_content_embed = comment_embeddings[user_content_index, :]
    else:
        user_content_index = user_content['index']
        user_content_embed = np.mean(comment_embeddings[user_content_index, :], axis=0)

    interaction_data = {
        'viewed_content_user_id': str(viewed_content_user_id),
        'timestamp': int(interaction_timestamp),
        'edge_data': edge_data
    }

    prev_comments_df = user_comments_df.xs(slice(first_content_datetime, interaction_datetime), level='createtime').iloc[:-1]
    if not prev_comments_df.empty:
        if INSTANTANEOUS:
            prev_content = prev_comments_df.iloc[-1, :]
            prev_content_index = prev_content['index']
            prev_content_embed = comment_embeddings[prev_content_index, :]
        else:
            prev_content_index = prev_comments_df['index']
            prev_content_embeds = comment_embeddings[prev_content_index, :]
            prev_content_embed = np.mean(prev_content_embeds, axis=0)

        last_datetime = interaction_datetime - MAX_TIME_FRAME

        if edge_data['type'] == 'video_comment':
            viewed_content_index = assert_and_get_viewed_id(user_id, user_content, edge_data, time_videos_df, viewed_content_user_id, 'comment_id', 'video_id')
            viewed_content_embed = video_embeddings[viewed_content_index, :]

            # make sure time is exclusive exclusive
            global_content = time_videos_df.xs(slice(last_datetime, interaction_datetime), level='createtime').iloc[1:-1]

            if INSTANTANEOUS:
                norm_user_content_embed = comment_norm[user_content_index]
                norm_prev_content_embed = comment_norm[prev_content_index]
            else:
                norm_user_content_embed = np.linalg.norm(user_content_embed)
                norm_prev_content_embed = np.linalg.norm(prev_content_embed)

            interaction_data = process_global_data(time_videos_df, video_embeddings, last_datetime, interaction_datetime, user_content_embed, prev_content_embed, norm_user_content_embed, norm_prev_content_embed, interaction_data)

            prev_content_euclid_dist = np.linalg.norm(prev_content_embed - user_content_embed)
            viewed_content_euclid_dist = np.linalg.norm(viewed_content_embed - user_content_embed)
            viewed_prev_content_euclid_dist = np.linalg.norm(viewed_content_embed - prev_content_embed)

            prev_content_cosine_sim = np.dot(prev_content_embed, user_content_embed) / (norm_prev_content_embed * norm_user_content_embed)

            if INSTANTANEOUS:
                viewed_content_cosine_sim = video_comment_cosine_sim[viewed_content_index, user_content_index]
                viewed_prev_content_cosine_sim = video_comment_cosine_sim[viewed_content_index, prev_content_index]
            else:
                viewed_content_cosine_sim = np.dot(viewed_content_embed, user_content_embed) / (video_norm[viewed_content_index] * norm_user_content_embed)
                viewed_prev_content_cosine_sim = np.dot(viewed_content_embed, prev_content_embed) / (video_norm[viewed_content_index] * norm_prev_content_embed)

        elif edge_data['type'] == 'comment_reply':
            viewed_content_index = assert_and_get_viewed_id(user_id, user_content, edge_data, time_comments_df, viewed_content_user_id, 'comment_id_reply', 'comment_id')
            viewed_content_embed = comment_embeddings[viewed_content_index, :]

            global_content = time_comments_df.xs(slice(last_datetime, interaction_datetime), level='createtime').iloc[1:-1]

            if INSTANTANEOUS:
                norm_user_content_embed = comment_norm[user_content_index]
                norm_prev_content_embed = comment_norm[prev_content_index]
            else:
                norm_user_content_embed = np.linalg.norm(user_content_embed)
                norm_prev_content_embed = np.linalg.norm(prev_content_embed)

            interaction_data = process_global_data(time_comments_df, comment_embeddings, last_datetime, interaction_datetime, user_content_embed, prev_content_embed, norm_user_content_embed, norm_prev_content_embed, interaction_data)

            prev_content_euclid_dist = np.linalg.norm(prev_content_embed - user_content_embed)
            viewed_content_euclid_dist = np.linalg.norm(viewed_content_embed - user_content_embed)
            viewed_prev_content_euclid_dist = np.linalg.norm(viewed_content_embed - prev_content_embed)

            prev_content_cosine_sim = np.dot(prev_content_embed, user_content_embed) / (norm_prev_content_embed * norm_user_content_embed)

            viewed_content_cosine_sim = np.dot(viewed_content_embed, user_content_embed) / (comment_norm[viewed_content_index] * norm_user_content_embed)
            viewed_prev_content_cosine_sim = np.dot(viewed_content_embed, prev_content_embed) / (comment_norm[viewed_content_index] * norm_prev_content_embed)

        interaction_data['user_content_embed'] = [float(round(e, 8)) for e in user_content_embed]
        interaction_data['viewed_content_embed'] = [float(round(e, 8)) for e in viewed_content_embed]
        interaction_data['prev_content_embed'] = [float(round(e, 8)) for e in prev_content_embed]

        interaction_data['prev_user_content_euclid_dist'] = float(round(prev_content_euclid_dist, 4))
        interaction_data['prev_user_content_cosine_sim'] = float(round(prev_content_cosine_sim, 4))

        interaction_data['viewed_user_content_euclid_dist'] = float(round(viewed_content_euclid_dist, 4))
        interaction_data['viewed_user_content_cosine_sim'] = float(round(viewed_content_cosine_sim, 4))

        interaction_data['viewed_prev_content_euclid_dist'] = float(round(viewed_prev_content_euclid_dist, 4))
        interaction_data['viewed_prev_content_cosine_sim'] = float(round(viewed_prev_content_cosine_sim, 4))

        if not global_content.empty:
            get_neighbour_dists(interaction_data, global_content, neighbours, '', edge_data, video_embeddings, comment_embeddings, user_content_embed, user_content_index, prev_content_embed, prev_content_index, comment_norm)
            get_neighbour_dists(interaction_data, global_content, second_order_neighbours, 'second_', edge_data, video_embeddings, comment_embeddings, user_content_embed, user_content_index, prev_content_embed, prev_content_index, comment_norm)
            get_neighbour_dists(interaction_data, global_content, third_order_neighbours, 'third_', edge_data, video_embeddings, comment_embeddings, user_content_embed, user_content_index, prev_content_embed, prev_content_index, comment_norm)

    return interaction_data

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    graph_path = os.path.join(data_dir_path, 'graph_data.json')
    with open(graph_path, 'r') as f:
        node_link_data = json.load(f)

    multi_graph = nx.node_link_graph(node_link_data)
    multi_graph.remove_edges_from(nx.selfloop_edges(multi_graph))

    video_embeddings_path = os.path.join(data_dir_path, f'all_video_desc_bertweet_umap_{NUM_DIMS}_embeddings.npy')
    with open(video_embeddings_path, 'rb') as f:
        video_embeddings = np.load(f)

    video_desc_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    video_desc_df = pd.read_csv(video_desc_path, dtype={'video_id': str, 'author_id': str})
    video_desc_df = video_desc_df.loc[:, ~video_desc_df.columns.str.contains('^Unnamed')]
    video_desc_df['createtime'] = pd.to_datetime(video_desc_df['createtime'])
    time_videos_df = video_desc_df.reset_index().set_index(['createtime', 'author_id', 'video_id']).sort_index(level='createtime', ascending=True)

    comment_embeddings_path = os.path.join(data_dir_path, f'all_english_comment_bertweet_umap_{NUM_DIMS}_embeddings.npy')
    with open(comment_embeddings_path, 'rb') as f:
        comment_embeddings = np.load(f)

    comments_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    comments_df = pd.read_csv(comments_path, dtype={'comment_id': str, 'author_id': str})
    comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
    comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
    time_comments_df = comments_df.reset_index().set_index(['createtime', 'author_id', 'comment_id']).sort_index(level='createtime', ascending=True)

    # precompute things
    video_norm, comment_norm, video_comment_cosine_sim = precompute_mats(video_embeddings, comment_embeddings)

    user_seqs = {}
    nodes = list(multi_graph.nodes())
    for user_id in tqdm.tqdm(nodes[:USER_SAMPLE]):
        try:
            user_comments_df = time_comments_df.xs(user_id, level='author_id')
        except KeyError:
            continue
        except ValueError:
            raise

        # we only care for people with multiple content interactions so we can see diff over time
        if len(user_comments_df) < 2:
            continue

        first_content_datetime = user_comments_df.iloc[0].name[0]
        last_content_datetime = user_comments_df.iloc[-1].name[0]

        inter_items = []
        for u, v, edge_data in multi_graph.out_edges(user_id, data=True):
            if edge_data['type'] not in ['video_comment', 'comment_reply']:
                continue
            viewed_content_user_id = u if u != user_id else v
            inter_items.append((viewed_content_user_id, edge_data))

        # sort interactions by time
        inter_items = sorted(inter_items, key=lambda inter_item: inter_item[1]['unix_createtime'])

        neighbours, second_order_neighbours, third_order_neighbours = get_neighbours(multi_graph, user_id)

        user_seq = []
        for viewed_content_user_id, edge_data in inter_items:
            try:
                interaction_data = process_interaction(edge_data, user_comments_df, time_comments_df, time_videos_df, comment_embeddings, video_embeddings, first_content_datetime, last_content_datetime, viewed_content_user_id, video_norm, comment_norm, video_comment_cosine_sim, neighbours, second_order_neighbours, third_order_neighbours)
                user_seq.append(interaction_data)
            except KeyError:
                continue

        user_seqs[user_id] = user_seq

    if INSTANTANEOUS:
        file_name = f'user_seqs_instantaneous_{NUM_DIMS}.json'
    else:
        file_name = f'user_seqs_changepoint_{NUM_DIMS}.json'
    user_seq_path = os.path.join(data_dir_path, file_name)
    with open(user_seq_path, 'w') as f:
        json.dump(user_seqs, f)

if __name__ == '__main__':
    main()
