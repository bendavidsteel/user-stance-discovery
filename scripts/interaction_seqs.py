import datetime
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import tqdm

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    graph_path = os.path.join(data_dir_path, 'graph_data.json')
    with open(graph_path, 'r') as f:
        node_link_data = json.load(f)

    multi_graph = nx.node_link_graph(node_link_data)

    video_embeddings_path = os.path.join(data_dir_path, 'all_video_desc_twitter_roberta_embeddings.npy')
    with open(video_embeddings_path, 'rb') as f:
        video_embeddings = np.load(f)

    video_desc_path = os.path.join(data_dir_path, 'all_video_desc.csv')
    video_desc_df = pd.read_csv(video_desc_path, dtype={'video_id': str, 'author_id': str})
    video_desc_df['createtime'] = pd.to_datetime(video_desc_df['createtime'])
    time_videos_df = video_desc_df.reset_index().set_index('createtime').sort_index(ascending=True)
    users_video_df = video_desc_df.groupby('author_id')

    comment_embeddings_path = os.path.join(data_dir_path, 'all_english_comment_twitter_roberta_embeddings.npy')
    with open(comment_embeddings_path, 'rb') as f:
        comment_embeddings = np.load(f)

    comments_path = os.path.join(data_dir_path, 'all_english_comments.csv')
    comments_df = pd.read_csv(comments_path, dtype={'comment_id': str, 'author_id': str})
    comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
    time_comments_df = comments_df.reset_index().set_index('createtime').sort_index(ascending=True)
    users_comment_df = comments_df.groupby('author_id')

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

        # get second order neighbours
        second_order_neighbours = set()
        for viewed_content_user_id, _ in inter_items:
            second_order_neighbours.update(multi_graph.successors(viewed_content_user_id))
            second_order_neighbours.update(multi_graph.predecessors(viewed_content_user_id))

        # make sure they're only second order neighbours
        second_order_neighbours.remove(user_id)
        for viewed_content_user_id, _ in inter_items:
            if viewed_content_user_id in second_order_neighbours:
                second_order_neighbours.remove(viewed_content_user_id)

        user_seq = []
        user_content_embeds = []
        for viewed_content_user_id, edge_data in inter_items:
            interaction_timestamp = edge_data['unix_createtime']
            interaction_datetime = datetime.datetime.fromtimestamp(interaction_timestamp)
            last_timestamp = user_seq[-1]['interaction_timestamp'] if len(user_seq) > 0 else 0
            last_datetime = datetime.datetime.fromtimestamp(last_timestamp)

            if edge_data['type'] == 'video_comment':
                try:
                    user_videos_df = users_video_df.get_group(viewed_content_user_id)
                    user_comments_df = users_comment_df.get_group(user_id)
                except KeyError:
                    continue

                viewed_content = user_videos_df[user_videos_df['video_id'] == edge_data['video_id']]
                user_content = user_comments_df[user_comments_df['comment_id'] == edge_data['comment_id']]

                if len(viewed_content) != 1 or len(user_content) != 1:
                    continue

                global_content = time_videos_df.loc[last_datetime:interaction_datetime]
                second_order_neighbour_content = global_content[global_content['author_id'].isin(second_order_neighbours)]

                global_content_embed = video_embeddings[global_content['index'].to_numpy(), :]
                second_order_neighbour_content_embed = video_embeddings[second_order_neighbour_content['index'].to_numpy(), :]
                viewed_content_embed = video_embeddings[viewed_content.index[0], :]
                user_content_embed = comment_embeddings[user_content.index[0], :]
            elif edge_data['type'] == 'comment_reply':
                try:
                    viewed_user_comments_df = users_comment_df.get_group(viewed_content_user_id)
                    user_comments_df = users_comment_df.get_group(user_id)
                except KeyError:
                    continue

                viewed_content = viewed_user_comments_df[viewed_user_comments_df['comment_id'] == edge_data['comment_id']]
                user_content = user_comments_df[user_comments_df['comment_id'] == edge_data['comment_id_reply']]

                if len(viewed_content) != 1 or len(user_content) != 1:
                    continue

                global_content = time_comments_df.loc[last_datetime:interaction_datetime]
                second_order_neighbour_content = global_content[global_content['author_id'].isin(second_order_neighbours)]

                global_content_embed = comment_embeddings[global_content['index'].to_numpy(), :]
                second_order_neighbour_content_embed = comment_embeddings[second_order_neighbour_content['index'].to_numpy(), :]
                viewed_content_embed = comment_embeddings[viewed_content.index[0], :]
                user_content_embed = comment_embeddings[user_content.index[0], :]
            else:
                continue

            interaction_data = {
                'viewed_content_user_id': str(viewed_content_user_id),
                'interaction_timestamp': int(interaction_timestamp)
            }

            if len(user_seq) > 0:
                prev_content_embed = user_content_embeds[-1]

                #TODO try cosine similarity instead of euclidean distance

                prev_content_dist = np.linalg.norm(user_content_embed - prev_content_embed)

                viewed_content_dist = np.linalg.norm(viewed_content_embed - user_content_embed)
                viewed_content_dist_share = prev_content_dist / (prev_content_dist + viewed_content_dist)

                interaction_data['viewed_content_dist_share'] = float(viewed_content_dist_share)

                # TODO ensure exclude viewed from global and all lower tiers from higher tiers

                global_content_dists = np.linalg.norm(global_content_embed - user_content_embed, axis=1)
                global_content_dist_shares = prev_content_dist / (prev_content_dist + global_content_dists)

                interaction_data['global_content_dist_shares'] = [float(share) for share in global_content_dist_shares]

                second_order_neighbour_content_dists = np.linalg.norm(second_order_neighbour_content_embed - user_content_embed, axis=1)
                second_order_neighbour_content_dist_shares = prev_content_dist / (prev_content_dist + second_order_neighbour_content_dists)

                interaction_data['second_order_neighbour_content_dist_shares'] = [float(share) for share in second_order_neighbour_content_dist_shares]

            user_seq.append(interaction_data)
            user_content_embeds.append(user_content_embed)

        user_seqs[user_id] = user_seq

    user_seq_path = os.path.join(data_dir_path, 'user_seqs.json')
    with open(user_seq_path, 'w') as f:
        json.dump(user_seqs, f)

if __name__ == '__main__':
    main()