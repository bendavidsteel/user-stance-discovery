import json
import os
import re

import numpy as np
import pandas as pd
import polars as pl
import torch
import tqdm

import generative
import opinions

class SocialInteractionDataset(torch.utils.data.Dataset):
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


class GenerativeInteractionDataset(SocialInteractionDataset):
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


class TikTokInteractionDataset(SocialInteractionDataset):
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


class RedditInteractionDataset(SocialInteractionDataset):
    def __init__(self):
        pass


class OpinionTimelineDataset:
    def __init__(self, comment_df, stance_columns, all_classifier_profiles, aggregation=None, halflife=50.):
        assert 'createtime' in comment_df.columns, "createtime column not found"
        assert 'user_id' in comment_df.columns, "user_id column not found"
        assert 'comment_id' in comment_df.columns, "comment_id column not found"
        assert 'comment' in comment_df.columns, "comment column not found"
        assert 'classifier_idx' in comment_df.columns, "classifier_idx column not found"
        self.comment_df = comment_df
        self.num_opinions = len(stance_columns)
        self.stance_columns = stance_columns
        self.all_classifier_profiles = all_classifier_profiles
        self.num_people = len(self.comment_df['user_id'].unique())
        self.aggregation = aggregation
        if self.aggregation == "sliding_exp":
            self.max_time_step = self.comment_df['createtime'].max()
        elif self.aggregation == "mean":
            self.max_time_step = 1
        else:
            self.max_time_step = self.comment_df['createtime'].max()

        self.aggregation = aggregation
        self.halflife = halflife

    def __getitem__(self, time_idx):
        comment_df = self.comment_df[self.comment_df['createtime'] <= time_idx]

        if len(comment_df) == 0:
            return np.zeros((0, self.num_opinions)), np.zeros((0, self.num_opinions))

        user_comments_df = comment_df.groupby('user_id')
        max_content = user_comments_df.count()['comment_id'].max()
        users = user_comments_df.count().index.values

        opinion_sequences = np.zeros((len(users), max_content, self.num_opinions))
        sequence_mask = np.zeros((len(users), max_content))
        classifier_indices = np.zeros((len(users), max_content), dtype=np.int64)

        for i, (user_id, user_comments) in enumerate(user_comments_df):
            user_comments = user_comments.sort_values('createtime')
            user_opinions = user_comments[self.stance_columns].values
            opinion_sequences[i, :len(user_opinions), :] = user_opinions
            sequence_mask[i, :len(user_opinions)] = 1
            classifier_indices[i, :len(user_opinions)] = user_comments['classifier_idx'].values

        if self.aggregation == "mean":
            opinion_snapshots = np.nanmean(opinion_sequences, axis=1)
            return opinion_snapshots, users
        elif self.aggregation == "weighted_mean":
            opinion_means, opinion_variances = self._weighted_statistics(opinion_sequences, sequence_mask, classifier_indices)
            return opinion_means, opinion_variances, users
        elif self.aggregation == "sliding_exp":
            raise NotImplementedError("TODO: Account for nans in data")
            opinion_snapshots = opinions.sliding_window(opinion_sequences, sequence_mask, halflife=self.halflife)
            return opinion_snapshots, users
        else:
            return opinion_sequences, sequence_mask, users, classifier_indices

    def get_user_comments(self, user_id):
        return self.comment_df[self.comment_df['user_id'] == user_id]
    
    def _weighted_statistics(self, opinion_sequences, mask_sequence, classifier_indices):
        stances = [col.replace('stance_', '') for col in self.stance_columns]
        precisions = np.zeros((len(stances), max([len(self.all_classifier_profiles[stance]) for stance in stances]), 3))
        precisions = np.nan * precisions
        for stance_idx, stance in enumerate(stances):
            for predictor_idx, predictor_id in enumerate(self.all_classifier_profiles[stance]):
                profile = self.all_classifier_profiles[stance][predictor_id]
                precisions[stance_idx, predictor_idx, 0] = profile['against']['precision']
                precisions[stance_idx, predictor_idx, 1] = profile['neutral']['precision']
                precisions[stance_idx, predictor_idx, 2] = profile['favor']['precision']

        weights = np.full(opinion_sequences.shape, np.nan)
        for stance_idx, stance in enumerate(stances):
            for user_idx in range(opinion_sequences.shape[0]):
                # Identify valid comments for the current user and stance
                valid_comments = (mask_sequence[user_idx, :] != 0) & (~np.isnan(opinion_sequences[user_idx, :, stance_idx]))

                # Get the predictor IDs for the current user across all comments
                predictor_ids = classifier_indices[user_idx, :]

                # Get the predicted stances for valid comments and adjust for indexing
                predicted_stances = opinion_sequences[user_idx, valid_comments, stance_idx].astype(int) + 1

                # Calculate precision values for valid comments
                precisions_values = precisions[stance_idx, predictor_ids[valid_comments], predicted_stances]

                # Update weights for valid comments
                weights[user_idx, valid_comments, stance_idx] = precisions_values


        def nansum(x, **kwargs):
            if np.isnan(x).all():
                return np.nan
            else:
                return np.nansum(x, **kwargs)

        weight_sum = nansum(weights, axis=1)

        weighted_mean = np.divide(nansum(opinion_sequences * weights, axis=1), weight_sum, out=np.full(weight_sum.shape, np.nan), where=weight_sum != 0)
        ex2 = np.divide(nansum(opinion_sequences ** 2 * weights, axis=1), weight_sum, out=np.full(weight_sum.shape, np.nan), where=weight_sum != 0)
        n = nansum(np.isnan(opinion_sequences), axis=1)
        weighted_variance = (ex2 - (weighted_mean ** 2)) * np.divide(n, n - 1, out=np.full(n.shape, np.nan), where=n > 1)
        return weighted_mean, weighted_variance

class GenerativeOpinionTimelineDataset(OpinionTimelineDataset):
    def __init__(self, num_people, max_time_step, num_opinions, **kwargs):
        generative_model = generative.SocialGenerativeModel(
            num_users=num_people,
            num_opinions=num_opinions,
        )

        time_step = 0
        generative_model.reset()

        for time_step in tqdm.tqdm(range(max_time_step)):
            generative_model.update(time_step)

        comments_df = generative_model.platform.comments
        posts_df = generative_model.platform.posts
        user_df = generative_model.users

        if len(comments_df) == 0:
            raise ValueError("No comments generated")

        comments_df = comments_df.reset_index()
        posts_df = posts_df.reset_index()
        user_df = user_df.reset_index()
        user_df = user_df[['index']]

        comments_df.columns = ['comment_id', 'opinions', 'post', 'comment', 'createtime', 'user_id', 'post_id', 'parent_comment_id']
        posts_df.columns = ['post_id', 'opinions', 'createtime', 'user_id']
        user_df.columns = ['user_id']

        num_opinions = generative_model.num_opinions

        stance_columns = [f'stance_{i}' for i in range(num_opinions)]
        comments_df[stance_columns] = np.array([np.array(o) for o in comments_df['opinions']])

        all_classifier_profiles = {}
        for stance_idx in range(num_opinions):
            all_classifier_profiles[stance_idx] = {}
            all_classifier_profiles[stance_idx][0] = {
                'true_favor': {
                    'predicted_favor': 1,
                    'predicted_against': 0,
                    'predicted_neutral': 0,
                },
                'true_against': {
                    'predicted_favor': 0,
                    'predicted_against': 1,
                    'predicted_neutral': 0,
                },
                'true_neutral': {
                    'predicted_favor': 0,
                    'predicted_against': 0,
                    'predicted_neutral': 1,
                },
            }

        super().__init__(comments_df, stance_columns, all_classifier_profiles, **kwargs)


class SimpleGenerativeOpinionTimelineDataset(OpinionTimelineDataset):
    def __init__(self, user_stance, user_stance_variance, num_data_points, prediction_precision=1.0, **kwargs):
        num_opinions = 1
        stance = 'stance_0'
        user_id = 'user0'

        comment_data = []
        for i in range(num_data_points):
            comment_stance = np.random.normal(user_stance, user_stance_variance)
            comment_stance = np.clip(comment_stance, -1, 1)
            quantized_comment_stance = 0
            if comment_stance > 1/3:
                quantized_comment_stance = 1
            elif comment_stance < -1/3:
                quantized_comment_stance = -1
            comment_data.append({
                'comment_id': f'comment{i}',
                'post': f'post{i}',
                'comment': f'comment{i}',
                'createtime': i,
                'user_id': user_id,
                'post_id': f'post{i}',
                'parent_comment_id': None,
                stance: quantized_comment_stance,
                'classifier_idx': 0,
            })
        comments_df = pd.DataFrame(comment_data)

        stance_columns = [stance]

        all_classifier_profiles = {}
        all_classifier_profiles[stance] = {}
        all_classifier_profiles[stance][0] = {
            'true_favor': {
                'predicted_favor': int(100 * prediction_precision),
                'predicted_against': 0,
                'predicted_neutral': int(100 * (1 - prediction_precision)) + int(100 * (1 - prediction_precision) / 2.),
            },
            'true_against': {
                'predicted_favor': 0,
                'predicted_against': int(100 * prediction_precision),
                'predicted_neutral': int(100 * (1 - prediction_precision)) + int(100 * (1 - prediction_precision) / 2.),
            },
            'true_neutral': {
                'predicted_favor': int(100 * (1 - prediction_precision)),
                'predicted_against': int(100 * (1 - prediction_precision)),
                'predicted_neutral': int(100 * prediction_precision),
            },
        }

        super().__init__(comments_df, stance_columns, all_classifier_profiles, **kwargs)


class EnsembleClassifier:
    def __init__(self, metric_profiles):
        self.metric_profiles = metric_profiles

    def predict(self, data_row):

        # TODO change to bayesian model averaging
        # for now we assume that neutral is the most common stance
        # and therefore anything except neutral should be prioritized if the classifier has a high precision for that label
        best_stance = 'neutral'
        highest_precision = 0
        best_classifier_idx = None
        for classifier_idx in self.metric_profiles:
            classification = data_row[f'stance_{classifier_idx}']
            metric_profile = self.metric_profiles[classifier_idx]
            classifier_precision = metric_profile[classification]['precision']
            if classifier_precision > highest_precision:
                highest_precision = classifier_precision
                best_classifier_idx = classifier_idx

                if classification != 'neutral' and classifier_precision > 0.5:
                    best_stance = classification

        return best_stance, best_classifier_idx

class RedditOpinionTimelineDataset(OpinionTimelineDataset):
    def __init__(self, **kwargs):

        this_dir_path = os.path.dirname(os.path.realpath(__file__))
        root_dir_path = os.path.join(this_dir_path, "..", "..")
        topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "1sub_1year", "topics_minilm_0_2")
        with open(os.path.join(topics_dir_path, "topic_stances.json"), "r") as f:
            topics_stances = json.load(f)

        comments_df = None
        all_metric_profiles = {}
        for topic_stances in topics_stances['topic_stances']:
            topic_path = os.path.join(topics_dir_path, f"topic_{topic_stances['topics'][0]}")
            for stance in topic_stances['stances']:
                stance_slug = stance.replace(" ", "_")
                stance_path = os.path.join(topic_path, stance_slug)

                if not os.path.exists(stance_path):
                    continue
                
                stance_comments_df = None

                metric_profiles = {}
                for idx, stance_focus in enumerate(['favor', 'against', 'neutral']):
                    best_metric = 0
                    best_metrics = None
                    best_dir_name = None
                    for dir_name in os.listdir(stance_path):
                        metrics_path = os.path.join(stance_path, dir_name, "metrics.json")
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                        metric = metrics[stance_focus]['f0.5']
                        if metric > best_metric and metrics[stance_focus]['precision'] > 0.5:
                            best_metric = metric
                            best_metrics = metrics
                            best_dir_name = dir_name
                    
                    if best_dir_name is None:
                        continue

                    pred_path = os.path.join(stance_path, best_dir_name)
                    # get smallest stance predictions
                    num_comments = None
                    file_name = None
                    for stance_preds_file in os.listdir(pred_path):
                        if stance_preds_file.endswith('comments.parquet.zstd'):
                            this_num_comments = re.search(r'\d+', stance_preds_file).group()
                            if num_comments is None or this_num_comments < num_comments:
                                num_comments = this_num_comments
                                file_name = stance_preds_file

                    if file_name is None or not os.path.exists(os.path.join(pred_path, file_name)):
                        continue

                    focus_stance_comments_df = pl.read_parquet(
                        os.path.join(pred_path, file_name), 
                        columns=['author', 'id', 'created_utc', 'body', 'body_parent', 'post_all_text', 'stance']
                    )
                    
                    # rename stance column to stance_{stance}
                    focus_stance_comments_df = focus_stance_comments_df.rename(
                        {
                            f"stance": f"stance_{idx}",
                            "author": "user_id",
                            "id": "comment_id",
                            "created_utc": "createtime",
                            "body": "comment",
                            "body_parent": "parent_comment",
                            "post_all_text": "post",
                        }
                    )

                    metric_profiles[idx] = best_metrics

                    if stance_comments_df is None:
                        stance_comments_df = focus_stance_comments_df
                    else:
                        stance_comments_df = stance_comments_df.join(focus_stance_comments_df.select(['comment_id', f'stance_{idx}']), on='comment_id', how='inner')

                # find best stance prediction for each comment based on metrics for each predictor
                # use a weighted voting method
                stance_comments_df = stance_comments_df.to_pandas()
                weighted_voting_classifier = EnsembleClassifier(metric_profiles)
                stance_comments_df[[f'stance_{stance_slug}', 'classifier_idx']] = stance_comments_df.apply(weighted_voting_classifier.predict, axis=1, result_type='expand')
                stance_comments_df = pl.from_pandas(stance_comments_df.drop([f'stance_{idx}' for idx in metric_profiles], axis=1))

                if comments_df is None:
                    comments_df = stance_comments_df
                else:
                    comments_df = pl.concat([comments_df, stance_comments_df], how='diagonal')    

                all_metric_profiles[stance_slug] = metric_profiles

        stance_columns = [col for col in comments_df.columns if "stance" in col]

        comments_df = comments_df.to_pandas()

        # map neutral to 0, favor to 1, against to -1
        for stance_column in stance_columns:
            def map_stance(stance):
                if stance == "neutral":
                    return 0
                elif stance == "favor":
                    return 1
                elif stance == "against":
                    return -1
                else:
                    return None
            comments_df[stance_column] = comments_df[stance_column].apply(map_stance)

        # round createtime to days, and rescale to start at 0
        comments_df['createtime'] = pd.to_datetime(comments_df['createtime'], unit='s')
        comments_df['createtime'] = comments_df['createtime'].dt.floor('d')
        comments_df['createtime'] = (comments_df['createtime'] - comments_df['createtime'].min()).dt.days

        super().__init__(comments_df, stance_columns, all_metric_profiles, **kwargs)
