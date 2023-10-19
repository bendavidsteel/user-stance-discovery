import os

import numpy as np
import pandas as pd
import torch
import tqdm

class SocialDataset(torch.utils.data.Dataset):
    def __init__(self, user_df, posts_df, comments_df, post_opinions, comment_opinions, batch=True):
        self.batch = batch

        self.user_df = user_df
        self.posts_df = posts_df
        self.comments_df = comments_df
        self.post_opinions = post_opinions
        self.comment_opinions = comment_opinions

        self.posts_comments_df = self.comments_df.sort_values(['post_id', 'createtime']).groupby('post_id')
        self.user_comments_df = self.comments_df.sort_values(['author_id', 'createtime']).groupby('author_id')

        self.users_df = self.comments_df[['author_id', 'author_name']].drop_duplicates().reset_index()

    def __len__(self):
        return len(self.users_df)

    def __getitem__(self, idx):
        users = self.users_df.iloc[idx]
        comments = pd.concat([self.user_comments_df.get_group(user_id) for user_id in users['author_id']])
        post_ids = comments['post_id'].unique()
        post_comments = pd.concat([self.posts_comments_df.get_group(post_id) for post_id in post_ids])
        # get only level 1 comments
        base_comments = post_comments[post_comments['reply_comment_id'].isna()]
        comment_opinions = self.comment_opinions[base_comments.index.values]

        if self.batch:
            num_users = len(users)
            max_comment_seq = comments[['author_id', 'comment_id']].groupby('author_id').count().max().values[0]
            max_base_comments = base_comments[['post_id', 'comment_id']].groupby('post_id').count().max().values[0]
            
            # add post opinions in front of comment opinions as default option for replying to a post
            padded_reply_opinions = np.zeros((num_users, max_comment_seq, max_base_comments + 1, comment_opinions.shape[1]))
            mask_reply_opinions = np.zeros((num_users, max_comment_seq, max_base_comments + 1,))
            chosen_reply_idx = np.zeros((num_users, max_comment_seq), dtype=np.int64)
            user_comment_opinions = np.zeros((num_users, max_comment_seq, comment_opinions.shape[1]))

            for i, user_id in enumerate(users['author_id']):
                comments = self.user_comments_df.get_group(user_id)
                for j, (comment_idx, comment) in enumerate(comments.iterrows()):
                    # get post
                    post = self.posts_df[self.posts_df['post_id'] == comment['post_id']]
                    assert len(post) == 1
                    post = post.iloc[0]
                    post_opinions = self.post_opinions[post['post_id']]

                    # get all comments for post
                    post_comments = self.posts_comments_df.get_group(comment['post_id'])
                    base_comments = post_comments[post_comments['reply_comment_id'].isna()]
                    comment_opinions = self.comment_opinions[base_comments.index.values]

                    # fill in the comment opinions
                    padded_reply_opinions[i, j, 0, :] = post_opinions
                    padded_reply_opinions[i, j, 1:comment_opinions.shape[0]+1, :] = comment_opinions
                    mask_reply_opinions[i, j, :comment_opinions.shape[0]+1] = 1

                    # fill in the reply comment
                    if not isinstance(comment['reply_comment_id'], str) and np.isnan(comment['reply_comment_id']):
                        reply_comment_idx = 0
                    else:
                        reply_comments = post_comments[post_comments['comment_id'] == comment['reply_comment_id']]
                        assert len(reply_comments) == 1
                        all_comments_reply_idx = reply_comments.index.values[0]
                        # adding 1 because we append the null comment zero vector at the start
                        reply_comment_idx = np.where(base_comments.index.values == all_comments_reply_idx)[0][0] + 1

                    chosen_reply_idx[i, j] = reply_comment_idx

                    # fill in the actual comment
                    user_comment_opinions[i, j, :] = self.comment_opinions[comment_idx]
        else:
            raise NotImplementedError()
            padded_reply_opinions = np.zeros((len(base_comments) + 1, comment_opinions.shape[1]))
            padded_reply_opinions[1:comment_opinions.shape[0]+1,:] = comment_opinions
            mask_reply_opinions = np.ones((len(base_comments) + 1,))

        return {
            'padded_reply_opinions': torch.tensor(padded_reply_opinions),
            'mask_reply_opinions': torch.tensor(mask_reply_opinions),
            'chosen_reply_idx': torch.tensor(chosen_reply_idx),
            'user_comment_opinions': torch.tensor(user_comment_opinions),
        }


class GenerativeDataset(SocialDataset):
    def __init__(self, model_creator, num_people=1000, max_time_step=1000):
        generative_model = model_creator(num_users=num_people)

        time_step = 0
        generative_model.reset()

        for time_step in tqdm.tqdm(range(max_time_step)):
            generative_model.update(time_step)

        comments_df = generative_model.platform.comments
        posts_df = generative_model.platform.posts
        user_df = generative_model.users

        comments_df = comments_df.reset_index()
        posts_df = posts_df.reset_index()
        user_df = user_df.reset_index()
        user_df = user_df[['index']]

        comments_df.columns = ['comment_id', 'opinions', 'post_id', 'reply_comment_id', 'createtime']
        posts_df.columns = ['post_id', 'opinions', 'createtime']
        user_df.columns = ['author_id']

        post_opinions = generative_model.platform.get_posts_positions()
        comment_opinions = generative_model.platform.get_comments_positions()

        super().__init__(user_df, posts_df, comments_df, post_opinions, comment_opinions)


class TikTokInteractionDataset(SocialDataset):
    def __init__(self, data_path, batch=True):
        self.batch = batch

        comments_df = pd.read_csv(os.path.join(data_path, 'sample_comments.csv'), dtype={'author_name': str, 'author_id': str, 'comment_id': str, 'video_id': str, 'reply_comment_id': str})
        # fix annoying reply_comment_id stored as exponent format
        comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
        comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
        
        videos_df = pd.read_csv(os.path.join(data_path, 'sample_videos.csv'), dtype={'author_name': str, 'author_id': str, 'video_id': str, 'share_video_id': str, 'share_video_user_id': str})
        videos_df = videos_df.loc[:, ~videos_df.columns.str.contains('^Unnamed')]
        videos_df['createtime'] = pd.to_datetime(videos_df['createtime'])

        users_df = pd.read_csv(os.path.join(data_path, 'sample_users.csv'), dtype={'author_name': str, 'author_id': str})
        
        # TODO switch from topic probs to embedding
        topic_probs = np.load(os.path.join(data_path, 'probs.npy'))
        comments_topic_probs = topic_probs[:len(comments_df), :]
        video_topic_probs = topic_probs[len(comments_df):, :]

        # get rid of comments where we don't have the replied comment
        comments_df = comments_df[(comments_df['reply_comment_id'].isna()) | (comments_df['reply_comment_id'].isin(comments_df['comment_id'].values))]
        comments_topic_probs = comments_topic_probs[comments_df.index.values]

        video_topic_probs = video_topic_probs[videos_df.index.values]

        comments_df = comments_df.reset_index()
        videos_df = videos_df.reset_index()

        self.__init__(users_df, videos_df, comments_df, video_topic_probs, comments_topic_probs)


class RedditInteractionDataset(SocialDataset):
    def __init__():
        pass