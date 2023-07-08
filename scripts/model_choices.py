import logging
import os

import torch
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyro.distributions.constraints as constraints
from pyro.generic import distributions as dist
from pyro.generic import infer, ops, optim, pyro, pyro_backend

import opinions

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, batch=True):
        self.batch = batch

        comments_df = pd.read_csv(os.path.join(data_path, 'sample_comments.csv'), dtype={'author_name': str, 'author_id': str, 'comment_id': str, 'video_id': str, 'reply_comment_id': str})
        # fix annoying reply_comment_id stored as exponent format
        comments_df = comments_df.loc[:, ~comments_df.columns.str.contains('^Unnamed')]
        comments_df['createtime'] = pd.to_datetime(comments_df['createtime'])
        
        videos_df = pd.read_csv(os.path.join(data_path, 'sample_videos.csv'), dtype={'author_name': str, 'author_id': str, 'video_id': str, 'share_video_id': str, 'share_video_user_id': str})
        videos_df = videos_df.loc[:, ~videos_df.columns.str.contains('^Unnamed')]
        videos_df['createtime'] = pd.to_datetime(videos_df['createtime'])
        
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


def uniform_probs(num):
    return torch.full((num,), 1 / num)

def uniform_probs_from_mask(mat_mask):
    vec_mask = mat_mask.any(dim=1)
    uniform_probs = torch.zeros(mat_mask.shape, dtype=torch.float64)
    uniform_probs[vec_mask] = mat_mask[vec_mask] / torch.sum(mat_mask, dim=1, keepdim=True)[vec_mask]
    return uniform_probs


def recommend_potential_videos(available_videos, user_history):
    probs = uniform_probs(len(available_videos))
    # TODO weight up if similar to videos previously consumed
    # TODO weight up if from author previously consumed
    return probs

def state_update(user_state, new_content, attention):
    return opinions.degroot(user_state, new_content, attention)

def get_video_opinions(video):
    return video.topics

def get_comments_opinions(comments):
    return [comment.topic for comment in comments]

def get_choose_video_probs(recommended_video, user_state, diff_weight, n, avg_content):
    content_probs = opinions.stochastic_bounded_confidence_bernoulli(user_state, recommended_video, diff_weight, n, avg_content)
    # TODO add social probs
    # social_probs = None
    # return (social_coef * social_probs) + ((1 - social_coef) * content_probs)
    return content_probs

def choose_comment(comments, comment_mask, user_state, diff_weight):
    content_probs = opinions.stochastic_bounded_confidence_categorical(user_state, comments, comment_mask, diff_weight)
    # TODO add social probs
    # social_probs = None
    # return (social_coef * social_probs) + ((1 - social_coef) * content_probs)
    return content_probs

def write_comment_opinions(user_state):
    return user_state

def recommend_videos(available_videos, user_history, actual_video=None):
    # TODO any platform parameters must be sampled here, outside the plate
    # TODO this has implications for how we can subsample
    # tiktok recommends a video
    with pyro.plate("actions", num_data):
        video_probs = recommend_potential_videos(available_videos, user_history)
        recommended_video = pyro.sample("recommended_video", dist.Categorical(probs=video_probs), obs=actual_video)
        return recommended_video

def choose_video(recommended_video, user_state, avg_video, actual_chosen_video=None):
    with pyro.plate("actions", num_data):
        # TODO does user state update then choose to click on video? then whether or not we click on the video does not affect us choosing the video
        # user has state update from video
        video_opinions = get_video_opinions(recommended_video)
        # attention paid to the video
        video_attention = pyro.sample("video_attention", dist.Normal(0, 1))
        user_state = state_update(user_state, video_opinions, video_attention)

        # user chooses whether to comment
        # if this value is greater than 0, users with similar opinions are more likely to interact
        # if less than 0, users with differing opinions are more likely to interact
        # TODO maybe pick different prior
        video_diff_weight = pyro.sample("video_diff_weight", dist.Normal(0, 1))
        # is there even any point trying to estimate this?
        # number of videos user watches where we would expect them to comment on 1
        upper_lim_num_vids = 1000
        n = pyro.sample("num_videos_per_comment", dist.Categorical(probs=uniform_probs(upper_lim_num_vids)))
        choose_video_probs = get_choose_video_probs(video_opinions, user_state, video_diff_weight, n, avg_video)
        chosen_video = pyro.sample("choose_video", dist.Bernoulli(choose_video_probs), obs=actual_chosen_video)
        return chosen_video, user_state

def choose_reply_comment(t, user_state, padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None):
    mask = mask_comment_opinions.any(axis=1)
    with pyro.poutine.mask(mask=mask):
        comment_attention = uniform_probs_from_mask(mask_comment_opinions)
        user_state = state_update(user_state, padded_comments_opinions, comment_attention)

        # select a comment to reply to
        # TODO maybe pick different prior
        comment_diff_weight_loc = pyro.param("comment_diff_weight_loc", init_tensor=torch.tensor(0.))
        comment_diff_weight_scale = pyro.param("comment_diff_weight_scale", init_tensor=torch.tensor(1.), constraint=constraints.positive)
        comment_diff_weight = pyro.sample(f"comment_diff_weight_{t}", dist.Normal(comment_diff_weight_loc, comment_diff_weight_scale))

        assert not comment_diff_weight.isnan().all(), "comment_diff_weight has nan values"
        comment_probs = choose_comment(padded_comments_opinions, mask_comment_opinions, user_state, comment_diff_weight)

        assert torch.logical_or(comment_probs[np.arange(comment_probs.shape[0]), actual_reply_comment] > 0, ~mask).all(), "Actual reply comment has 0 probability"
        # ensure simplex criteria is met for masked values
        comment_probs[~mask, 0] = 1
        chosen_comment = pyro.sample(f"choose_comment_{t}", dist.Categorical(probs=comment_probs).mask(mask), obs=actual_reply_comment)
        return chosen_comment, user_state

def choose_reply_comment_guide(user_state, padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None):
    with pyro.plate("choose_actions", len(padded_comments_opinions), dim=-1):
        for t in pyro.markov(range(padded_comments_opinions.shape[1])):
            # comments are then shown on video and causes state update
            # comment_attention = uniform_probs_from_mask(mask_comment_opinions)
            # user_state = state_update(user_state, padded_comments_opinions, comment_attention)

            # select a comment to reply to
            # TODO maybe pick different prior
            comment_diff_weight_loc = pyro.param("comment_diff_weight_loc_q", init_tensor=torch.tensor(0.), event_dim=1)
            pyro.sample(f"comment_diff_weight_{t}", dist.Delta(comment_diff_weight_loc))

def write_comment(t, user_state, actual_comment=None):
    # with pyro.plate("write_actions", user_state.shape[0], dim=-2):
    # write a comment
    comment_opinions = write_comment_opinions(user_state)
    comment_scale = pyro.param("comment_scale", torch.tensor(1.), constraint=constraints.positive)
    comment_scale = comment_scale.expand(comment_opinions.shape)
    comment_cov = torch.zeros((*comment_opinions.shape, 1), dtype=torch.float64)
    our_comment = pyro.sample(f"our_comment_{t}", dist.LowRankMultivariateNormal(comment_opinions, comment_cov, comment_scale), obs=actual_comment)
    return our_comment

def reply_and_write_comment(padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None, actual_comment=None):
    with pyro.plate("actions", padded_comments_opinions.shape[0]):
        user_state = torch.zeros((padded_comments_opinions.shape[0], padded_comments_opinions.shape[-1]))
        for t in pyro.markov(range(padded_comments_opinions.shape[1])):
            t_padded_comments_opinions = padded_comments_opinions[:, t]
            t_mask_comment_opinions = mask_comment_opinions[:, t]
            t_actual_reply_comment = actual_reply_comment[:, t]
            t_actual_comment = actual_comment[:, t]

            reply_comment, user_state = choose_reply_comment(t, user_state, t_padded_comments_opinions, t_mask_comment_opinions, actual_reply_comment=t_actual_reply_comment)
            our_comment = write_comment(t, user_state, actual_comment=t_actual_comment)
    return reply_comment, our_comment, user_state

def reply_and_write_comment_guide(padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None, actual_comment=None):
    choose_reply_comment_guide(padded_comments_opinions, mask_comment_opinions, actual_reply_comment)

def model(available_videos, user_history, actual_video=None, actual_chosen_video=None, actual_replied_comment=None, actual_comment=None):

    avg_video = torch.mean(available_videos)
    num_data = len(user_history)
    
    # TODO use user previous actions
    # TODO this is a markov process, use pyro.markov
    # if not user_state:
    user_state = pyro.sample("initial_user_state", dist.MultivariateNormal())
    recommended_video = recommend_videos(available_videos, user_history, video=actual_video)

    chosen_video, user_state = choose_video(recommended_video, user_state, avg_video, actual_chosen_video=actual_chosen_video)

    if chosen_video:
        reply_comment, user_state = choose_reply_comment(available_videos, recommended_video, actual_replied_comment=actual_replied_comment)

        comment = write_comment(user_state, actual_comment=actual_comment)

        return recommended_video, chosen_video, reply_comment, comment

def get_trainer(model_func, guide=None):
    if guide is None:
        guide = infer.autoguide.AutoDelta(model_func)
    adam = optim.ClippedAdam({"lr": 0.001})
    elbo = infer.Trace_ELBO()
    svi = infer.SVI(model_func, guide, adam, elbo)
    return svi

def training(dataloader, batch_size):

    backend = "pyro"
    with pyro_backend(backend):

        if backend == "pyro":
            pyro.enable_validation(True)
            pyro.set_rng_seed(1)

        trainer = get_trainer(reply_and_write_comment)#, guide=reply_and_write_comment_guide)

        num_epochs = 100

        for epoch in range(num_epochs):
            running_loss = 0.0
            num_preds = 0

            # Iterate over data.
            bar = tqdm.tqdm(
                dataloader,
                desc="Epoch {}".format(epoch),
            )
            for i, batch in enumerate(bar):
                padded_comments_opinions = batch['padded_comments_opinions']
                mask_comments_opinions = batch['mask_comments_opinions']
                reply_comment = batch['reply_comment']
                actual_comment = batch['actual_comment']

                loss = trainer.step(padded_comments_opinions, mask_comments_opinions, actual_reply_comment=reply_comment, actual_comment=actual_comment)

                # statistics
                running_loss += loss / batch_size
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.2f}".format(running_loss / num_preds),
                    )

            epoch_loss = running_loss / len(dataloader.dataset)

        alpha_q = pyro.param("alpha_q").item()
        beta_q = pyro.param("beta_q").item()

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data', 'bertweet_base_5_100_100')

    batch_size = 64
    num_workers = 0
    pin_memory = False
    shuffle = False

    dataset = InteractionDataset(data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False
        ),
        batch_size=None,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    training(dataloader, batch_size)

if __name__ == '__main__':
    main()