import logging
import os

import torch
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
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
        self.max_base_comments = self.comments_df[self.comments_df['reply_comment_id'].isna()][['video_id', 'comment_id']].groupby('video_id').count().max().values[0]

    def __len__(self):
        return len(self.comments_df)

    def __getitem__(self, idx):
        comment = self.comments_df.iloc[idx]
        video_comments = self.videos_comments_df.get_group(comment['video_id'])
        # get only level 1 comments
        base_comments = video_comments[video_comments['reply_comment_id'].isna()]
        comment_opinions = self.comments_topic_probs[base_comments.index.values]

        # add on 0s vec as null comment
        padded_comment_opinions = np.zeros((self.max_base_comments + 1, comment_opinions.shape[1]))
        padded_comment_opinions[1:comment_opinions.shape[0]+1,:] = comment_opinions
        mask_comment_opinions = np.zeros((self.max_base_comments + 1,))
        mask_comment_opinions[:comment_opinions.shape[0]+1] = 1

        if not isinstance(comment['reply_comment_id'], str) and np.isnan(comment['reply_comment_id']):
            reply_comment_idx = 0
        else:
            reply_comments = video_comments[video_comments['comment_id'] == comment['reply_comment_id']]
            assert len(reply_comments) == 1
            all_comments_reply_idx = reply_comments.index.values[0]
            # adding 1 because we append the null comment zero vector at the start
            reply_comment_idx = np.where(base_comments.index.values == all_comments_reply_idx)[0][0] + 1
        return {
            'user_state': torch.zeros(self.video_topic_probs.shape[1]),
            'padded_comments_opinions': torch.tensor(padded_comment_opinions),
            'mask_comments_opinions': torch.tensor(mask_comment_opinions),
            'reply_comment': torch.tensor(reply_comment_idx)
        }

def uniform_probs(num):
    return torch.full((num,), 1 / num)

def uniform_probs_from_mask(mask):
    return mask / torch.sum(mask, dim=1, keepdim=True)

def degroot(user_state, new_content, attention):
    return user_state + torch.bmm(attention.unsqueeze(1), new_content).squeeze(1)

def friedkin_johnsen(original_user_state, content, stubbornness):
    return ((1 - stubbornness) * original_user_state) + (stubbornness * torch.sum(content))

def bounded_confidence(user_state, new_content, confidence_interval):
    diff = user_state - new_content
    mask = torch.norm(torch.abs(diff), dim=1) > confidence_interval
    return user_state + ((1 / torch.sum(mask)) * torch.sum(mask * diff))

def stochastic_bounded_confidence_categorical(user_state, new_content, content_mask, exponent):
    # compares prob of interaction between a number of people
    unsq_content_mask = content_mask.unsqueeze(2)
    # we can't use cdist because we can't mask in cdist, and unmasked cdist would be computationally inefficient
    content_dist = torch.cdist(user_state.unsqueeze(1), new_content * unsq_content_mask).squeeze(1) * content_mask
    numerator = torch.pow(content_dist, -1 * exponent.unsqueeze(1)) * content_mask
    return numerator / torch.sum(numerator, dim=1, keepdims=True)

def stochastic_bounded_confidence_bernoulli(user_state, new_content, exponent, n, avg_content):
    # compares prob of interaction between a number of people
    numerator = torch.pow(torch.abs(user_state - new_content), -1 * exponent)
    return numerator / (numerator + ((n-1) * torch.pow(torch.abs(user_state - avg_content), -1 * exponent)))

def recommend_potential_videos(available_videos, user_history):
    probs = uniform_probs(len(available_videos))
    # TODO weight up if similar to videos previously consumed
    # TODO weight up if from author previously consumed
    return probs

def state_update(user_state, new_content, attention):
    return degroot(user_state, new_content, attention)

def get_video_opinions(video):
    return video.topics

def get_comments_opinions(comments):
    return [comment.topic for comment in comments]

def get_choose_video_probs(recommended_video, user_state, diff_weight, n, avg_content):
    content_probs = stochastic_bounded_confidence_bernoulli(user_state, recommended_video, diff_weight, n, avg_content)
    # TODO add social probs
    # social_probs = None
    # return (social_coef * social_probs) + ((1 - social_coef) * content_probs)
    return content_probs

def choose_comment(comments, comment_mask, user_state, diff_weight):
    content_probs = stochastic_bounded_confidence_categorical(user_state, comments, comment_mask, diff_weight)
    # TODO add social probs
    # social_probs = None
    # return (social_coef * social_probs) + ((1 - social_coef) * content_probs)
    return content_probs

def write_comment_opinions(user_state):
    return user_state

def recommend_videos(available_videos, user_history, actual_video=None):
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

def choose_reply_comment(user_state, padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None):
    with pyro.plate("actions", len(padded_comments_opinions)):
        # comments are then shown on video and causes state update
        comment_attention = uniform_probs_from_mask(mask_comment_opinions)
        user_state = state_update(user_state, padded_comments_opinions, comment_attention)

        # select a comment to reply to
        # TODO maybe pick different prior
        comment_diff_weight = pyro.sample("comment_diff_weight", dist.Normal(0, 1))
        comment_probs = choose_comment(padded_comments_opinions, mask_comment_opinions, user_state, comment_diff_weight)
        chosen_comment = pyro.sample("choose_comment", dist.Categorical(probs=comment_probs), obs=actual_reply_comment)
        return chosen_comment, user_state

def write_comment(user_state, actual_comment=None):
    with pyro.plate("actions", len(user_state)):
        # write a comment
        comment_opinions = write_comment_opinions(user_state)
        our_comment = pyro.sample("our_comment", dist.Normal(loc=comment_opinions, scale=1), obs=actual_comment)
        return our_comment

def model(available_videos, user_history, actual_video=None, actual_chosen_video=None, actual_replied_comment=None, actual_comment=None):

    avg_video = torch.mean(available_videos)
    num_data = len(user_history)
    
    # TODO use user previous actions
    # if not user_state:
    user_state = pyro.sample("initial_user_state", dist.MultivariateNormal())
    recommended_video = recommend_videos(available_videos, user_history, video=actual_video)

    chosen_video, user_state = choose_video(recommended_video, user_state, avg_video, actual_chosen_video=actual_chosen_video)

    if chosen_video:
        reply_comment, user_state = choose_reply_comment(available_videos, recommended_video, actual_replied_comment=actual_replied_comment)

        comment = write_comment(user_state, actual_comment=actual_comment)

        return recommended_video, chosen_video, reply_comment, comment

def get_trainer(model_func):
    auto_guide = pyro.infer.autoguide.AutoNormal(model_func)
    adam = pyro.optim.Adam({"lr": 0.02})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model_func, auto_guide, adam, elbo)
    return svi

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data', 'bertweet_base_5_100_100')

    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    inter_dataset = InteractionDataset(data_path)
    inter_dataloader = torch.utils.data.DataLoader(inter_dataset, batch_size=64, shuffle=True) #, num_workers=4)

    trainer = get_trainer(choose_reply_comment)

    num_epochs = 100

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_preds = 0

        # Iterate over data.
        bar = tqdm.tqdm(
            inter_dataloader,
            desc="Epoch {}".format(epoch),
        )
        for i, batch in enumerate(bar):
            user_state = batch['user_state']
            padded_comments_opinions = batch['padded_comments_opinions']
            mask_comments_opinions = batch['mask_comments_opinions']
            reply_comment = batch['reply_comment']

            loss = trainer.step(user_state, padded_comments_opinions, mask_comments_opinions, actual_reply_comment=reply_comment)

            # statistics
            running_loss += loss / inputs.size(0)
            num_preds += 1
            if i % 10 == 0:
                bar.set_postfix(
                    loss="{:.2f}".format(running_loss / num_preds),
                )

        epoch_loss = running_loss / len(inter_dataset)

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")

if __name__ == '__main__':
    main()