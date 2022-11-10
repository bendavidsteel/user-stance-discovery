from datetime import datetime
import json
import os
import re

import networkx as nx
import pandas as pd
import tqdm

import pytok

def to_jsonable_dict(row):
    d = {}
    for key, val in row.items():
        if isinstance(val, int) or isinstance(val, str) or val is None:
            json_val = val
        elif isinstance(val, list):
            json_val = ','.join(val)
        else:
            json_val = val
        d[key] = json_val

    return d

def add_edges_to_graph(df, u_id, v_id, edge_columns, edge_type, graph):
    time_cols = [edge_col for edge_col in edge_columns if 'createtime' in edge_col]
    if len(time_cols) == 1:
        time_col = time_cols[0]
        df['unix_createtime'] = df[time_col].map(pd.Timestamp.timestamp).astype(int)
        edge_columns.remove(time_col)
        edge_columns.append('unix_createtime')
    df['type'] = edge_type
    edge_columns.append('type')
    df['edge_data'] = df[edge_columns].apply(to_jsonable_dict, axis=1)
    edges_df = df[[u_id, v_id, 'edge_data']]
    edges = list(edges_df.itertuples(index=False, name=None))
    graph.add_edges_from(edges)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    comment_csv_path = os.path.join(data_dir_path, 'all_comments.csv')
    video_csv_path = os.path.join(data_dir_path, 'all_videos.csv')

    comment_df = pytok.utils.get_comment_df(comment_csv_path)
    video_df = pytok.utils.get_video_df(video_csv_path)

    count_comments_df = comment_df[['author_id', 'author_name', 'createtime', 'text', 'comment_language']].groupby(['author_id', 'author_name']).aggregate(list).reset_index()
    count_comments_df['comment_count'] = count_comments_df['createtime'].str.len()

    video_df = video_df.drop_duplicates('video_id')
    count_vids_df = video_df[['author_id', 'author_name', 'createtime', 'desc', 'hashtags']].groupby(['author_id', 'author_name']).aggregate(list).reset_index()
    count_vids_df['video_count'] = count_vids_df['createtime'].str.len()

    counts_df = count_vids_df.merge(count_comments_df, how='outer', on=['author_id', 'author_name']).fillna(0)
    counts_df[['video_count', 'comment_count']] = counts_df[['video_count', 'comment_count']].astype(int)

    interactions_df = video_df.rename(columns={'createtime': 'video_createtime', 'author_name': 'video_author_name', 'author_id': 'video_author_id', 'desc': 'video_desc', 'hashtags': 'video_hashtags'}) \
        .merge(comment_df.rename(columns={'createtime': 'comment_createtime', 'author_name': 'comment_author_name', 'author_id': 'comment_author_id', 'text': 'comment_text'}), on='video_id')

    mentions_df = comment_df[comment_df['mentions'].str.len() != 0][['author_id', 'mentions', 'text', 'createtime']].explode('mentions').drop_duplicates()
    mentions_df = mentions_df.rename(columns={'mentions': 'mention_id'})

    # add share edges
    shares_df = video_df[video_df['share_video_id'].notna()][['video_id', 'author_id', 'createtime', 'share_video_id', 'share_video_user_id', 'share_type']]

    # add video desc mentions
    video_mentions_df = video_df[video_df['mentions'].str.len() != 0][['video_id', 'author_id', 'createtime', 'mentions']] \
        .explode('mentions').rename(columns={'mentions': 'mention_id'})

    # add comment replies edges
    comment_replies_df = comment_df[comment_df['reply_comment_id'].notna()] \
        .merge(comment_df[comment_df['reply_comment_id'].isna()],
               left_on='reply_comment_id',
               right_on='comment_id',
               suffixes=('_reply', ''),
               how='left') \
        [['comment_id_reply', 'reply_comment_id_reply', 'author_id_reply', 'createtime_reply', 'comment_id', 'author_id']]

    user_ids = set(counts_df['author_id'].values)
    
    graph = nx.MultiDiGraph()

    graph.add_nodes_from(user_ids)

    # video comment replies
    add_edges_to_graph(interactions_df, 'comment_author_id', 'video_author_id', ['comment_createtime', 'video_id', 'video_desc', 'comment_id', 'comment_text'], 'video_comment', graph)

    # comment mentions
    add_edges_to_graph(mentions_df, 'author_id', 'mention_id', ['createtime', 'text'], 'comment_mention', graph)

    # video shares
    add_edges_to_graph(shares_df, 'author_id', 'share_video_user_id', ['createtime'], 'video_share', graph)

    # video_desc_mentions
    add_edges_to_graph(video_mentions_df, 'author_id', 'mention_id', ['createtime'], 'video_mention', graph)

    # comment replies
    add_edges_to_graph(comment_replies_df, 'author_id_reply', 'author_id', ['createtime_reply', 'comment_id_reply', 'comment_id'], 'comment_reply', graph)

    
    # write to file
    graph_data = nx.readwrite.node_link_data(graph)

    file_name = 'graph_data.json'

    graph_path = os.path.join(data_dir_path, file_name)
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f)

    print("Written new comment graph to file.")

if __name__ == '__main__':
    main()