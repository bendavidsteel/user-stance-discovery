import os

import numpy as np
import pandas as pd
import torch

class GenerativeDataset(torch.utils.data.Dataset):
    def __init__(self, model_creator):
        self.num_people = 1000
        self.model = model_creator(num_users=self.num_people)
        max_time_step = 1000

        self.comment_opinions = torch.zeros(())
        self.reply_comments = torch.zeros(())
        self.actual_comments = torch.zeros(())

        time_step = 0
        while time_step < max_time_step:
            self.model.update(time_step)

            self.comment_opinions[] = self.model.get
            self.reply_comments[] = self.model.get
            self.actual_comments[] = self.model.get

    def __len__(self):
        return self.num_people
    
    def __getitem__(self, idx):
        padded_comment_opinions = self.comment_opinions[idx]
        # TODO varying number of seen comments
        mask_comment_opinions = torch.ones()
        reply_comment = self.reply_comments[idx]
        actual_comment = self.actual_comments[idx]

        return {
            'padded_comments_opinions': torch.tensor(padded_comment_opinions),
            'mask_comments_opinions': torch.tensor(mask_comment_opinions),
            'reply_comment': torch.tensor(reply_comment),
            'actual_comment': torch.tensor(actual_comment),
        }


class TikTokInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, batch=True):
        self.batch = batch

        comments_df = pd.read_csv(os.path.join(data_path, 'sample_comments.csv'), dtype={'author_name': str, 'author_id': str, 'comment_id': str, 'video_id': str, 'reply_comment_id': str})
        # fix annoying reply_comment_id stored as exponent format
        comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
        comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
        
        videos_df = pd.read_csv(os.path.join(data_path, 'sample_videos.csv'), dtype={'author_name': str, 'author_id': str, 'video_id': str, 'share_video_id': str, 'share_video_user_id': str})
        videos_df = videos_df.loc[:, ~videos_df.columns.str.contains('^Unnamed')]
        videos_df['createtime'] = pd.to_datetime(videos_df['createtime'])
        
        # TODO switch from topic probs to embedding
        topic_probs = np.load(os.path.join(data_path, 'probs.npy'))
        comments_topic_probs = topic_probs[:len(comments_df), :]
        video_topic_probs = topic_probs[len(comments_df):, :]

        # get rid of comments where we don't have the replied comment
        comments_df = comments_df[(comments_df['reply_comment_id'].isna()) | (comments_df['reply_comment_id'].isin(comments_df['comment_id'].values))]
        self.comments_topic_probs = comments_topic_probs[comments_df.index.values]

        self.video_topic_probs = video_topic_probs[videos_df.index.values]

        self.comments_df = comments_df.reset_index()
        self.videos_df = videos_df.reset_index()

        self.videos_comments_df = self.comments_df.sort_values(['video_id', 'createtime']).groupby('video_id')
        self.user_comments_df = self.comments_df.sort_values(['author_id', 'createtime']).groupby('author_id')

        self.users_df = self.comments_df[['author_id', 'author_name']].drop_duplicates().reset_index()

    def __len__(self):
        return len(self.users_df)

    def __getitem__(self, idx):
        users = self.users_df.iloc[idx]
        comments = pd.concat([self.user_comments_df.get_group(user_id) for user_id in users['author_id']])
        video_ids = comments['video_id'].unique()
        video_comments = pd.concat([self.videos_comments_df.get_group(video_id) for video_id in video_ids])
        # get only level 1 comments
        base_comments = video_comments[video_comments['reply_comment_id'].isna()]
        comment_opinions = self.comments_topic_probs[base_comments.index.values]

        if self.batch:
            num_users = len(users)
            max_comment_seq = comments[['author_id', 'comment_id']].groupby('author_id').count().max().values[0]
            max_base_comments = base_comments[['video_id', 'comment_id']].groupby('video_id').count().max().values[0]
            
            # add on 0s vec as null comment
            raise Exception("Zero vec as null comment doesn't make sense, as embeddings are not necessarily centred around zero")
            padded_comment_opinions = np.zeros((num_users, max_comment_seq, max_base_comments + 1, comment_opinions.shape[1]))
            mask_comment_opinions = np.zeros((num_users, max_comment_seq, max_base_comments + 1,))
            reply_comment = np.zeros((num_users, max_comment_seq), dtype=np.int64)
            actual_comment = np.zeros((num_users, max_comment_seq, comment_opinions.shape[1]))

            for i, user_id in enumerate(users['author_id']):
                comments = self.user_comments_df.get_group(user_id)
                for j, (comment_idx, comment) in enumerate(comments.iterrows()):
                    # get all comments for video
                    video_comments = self.videos_comments_df.get_group(comment['video_id'])
                    base_comments = video_comments[video_comments['reply_comment_id'].isna()]
                    comment_opinions = self.comments_topic_probs[base_comments.index.values]

                    # fill in the comment opinions
                    padded_comment_opinions[i, j, 1:comment_opinions.shape[0]+1, :] = comment_opinions
                    mask_comment_opinions[i, j, :comment_opinions.shape[0]+1] = 1

                    # fill in the reply comment
                    if not isinstance(comment['reply_comment_id'], str) and np.isnan(comment['reply_comment_id']):
                        reply_comment_idx = 0
                    else:
                        reply_comments = video_comments[video_comments['comment_id'] == comment['reply_comment_id']]
                        assert len(reply_comments) == 1
                        all_comments_reply_idx = reply_comments.index.values[0]
                        # adding 1 because we append the null comment zero vector at the start
                        reply_comment_idx = np.where(base_comments.index.values == all_comments_reply_idx)[0][0] + 1

                    reply_comment[i, j] = reply_comment_idx

                    # fill in the actual comment
                    actual_comment[i, j, :] = self.comments_topic_probs[comment_idx]
        else:
            raise NotImplementedError()
            padded_comment_opinions = np.zeros((len(base_comments) + 1, comment_opinions.shape[1]))
            padded_comment_opinions[1:comment_opinions.shape[0]+1,:] = comment_opinions
            mask_comment_opinions = np.ones((len(base_comments) + 1,))

        return {
            'padded_comments_opinions': torch.tensor(padded_comment_opinions),
            'mask_comments_opinions': torch.tensor(mask_comment_opinions),
            'reply_comment': torch.tensor(reply_comment),
            'actual_comment': torch.tensor(actual_comment),
        }
    

class RedditInteractionDataset(torch.utils.data.Dataset):
    def __init__():
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass