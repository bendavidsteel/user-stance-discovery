import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

EPSILON = 1e-6

def movement_metric(prev_dist, target_dist):
    return prev_dist / (prev_dist + target_dist + EPSILON)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    user_seq_path = os.path.join(data_dir_path, 'user_seqs.json')
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    min_movement_video_comment = 1
    max_movement_video_comment = 0
    min_movement_comment_reply = 1
    max_movement_comment_reply = 0

    for user_id, user_seq in tqdm(user_seqs.items()):
        for inter in user_seq:

            # viewed_content_euclid_dist_share = prev_content_euclid_dist / (prev_content_euclid_dist + viewed_content_euclid_dist)
            # viewed_content_cosine_sim_share = prev_content_cosine_sim / (prev_content_cosine_sim + viewed_content_cosine_sim)

            # global_content_euclid_dist_share = prev_content_euclid_dist / (prev_content_euclid_dist + global_content_euclid_dist)
            # global_content_cosine_sim_share = prev_content_cosine_sim / (prev_content_cosine_sim + global_content_cosine_sim)

            # second_order_neighbour_content_euclid_dist_share = prev_content_euclid_dist / (prev_content_euclid_dist + second_order_neighbour_content_euclid_dist)
            # second_order_neighbour_content_cosine_sim_share = prev_content_cosine_sim / (prev_content_cosine_sim + second_order_neighbour_content_cosine_sim)

            inter_type = inter['type']
            viewed_movement = movement_metric(inter['prev_content_euclid_dist'], inter['viewed_content_euclid_dist'])

            if inter_type == 'video_comment':
                if viewed_movement > max_movement_video_comment:
                    max_movement_video_comment = viewed_movement
                    max_moved_video_comment = (user_id, inter)

                if viewed_movement < min_movement_video_comment:
                    min_movement_video_comment = viewed_movement
                    min_moved_video_comment = (user_id, inter)

            elif inter_type == 'comment_reply':
                if viewed_movement > max_movement_comment_reply:
                    max_movement_comment_reply = viewed_movement
                    max_moved_comment_reply = (user_id, inter)

                if viewed_movement < min_movement_comment_reply:
                    min_movement_comment_reply = viewed_movement
                    min_moved_comment_reply = (user_id, inter)

    video_desc_path = os.path.join(data_dir_path, 'old', 'all_video_desc.csv')
    video_desc_df = pd.read_csv(video_desc_path, dtype={'video_id': str, 'author_id': str})
    video_desc_df = video_desc_df.loc[:, ~video_desc_df.columns.str.contains('^Unnamed')]
    video_desc_df['createtime'] = pd.to_datetime(video_desc_df['createtime'])
    video_desc_df = video_desc_df.reset_index().set_index('video_id')
    video_desc_df = video_desc_df[~video_desc_df.index.duplicated(keep='first')]
    users_video_df = video_desc_df.groupby('author_id')

    comments_path = os.path.join(data_dir_path, 'old', 'all_english_comments.csv')
    comments_df = pd.read_csv(comments_path, dtype={'comment_id': str, 'author_id': str})
    comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
    comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
    comments_df = comments_df.reset_index().set_index('comment_id')
    comments_df = comments_df[~comments_df.index.duplicated(keep='first')]
    users_comment_df = comments_df.groupby('author_id')

    user_comments_df = users_comment_df.get_group(min_moved_video_comment[0])
    min_moved_comment = user_comments_df[user_comments_df['unix_createtime'] == min_moved_video_comment[1]['timestamp']]
    user_videos_df = users_video_df.get_group(min_moved_video_comment[1]['viewed_content_user_id'])
    min_moved_to_video = user_

if __name__ == '__main__':
    main()
