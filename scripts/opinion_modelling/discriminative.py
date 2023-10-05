import torch
import numpy as np

import pyro.distributions.constraints as constraints
from pyro.generic import distributions as dist
from pyro.generic import pyro

import opinions

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

class SocialDiscrimativeModel:
    def __init__(self, attend_to_comments='chosen'):
        self.attend_to_comments = attend_to_comments

    def choose_reply_comment(self, t, user_state, padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None):
        mask = mask_comment_opinions.any(axis=1)
        with pyro.poutine.mask(mask=mask):
            if self.attend_to_comments == 'all':
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

            if self.attend_to_comments == 'chosen':
                comment_attention = torch.zeros_like(comment_probs)
                comment_attention[np.arange(comment_probs.shape[0]), chosen_comment] = 1
                user_state = state_update(user_state, padded_comments_opinions, comment_attention)

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

    def reply_and_write_comment(self, padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None, actual_comment=None):
        with pyro.plate("actions", padded_comments_opinions.shape[0]):
            user_state = torch.zeros((padded_comments_opinions.shape[0], padded_comments_opinions.shape[-1]))
            for t in pyro.markov(range(padded_comments_opinions.shape[1])):
                t_padded_comments_opinions = padded_comments_opinions[:, t]
                t_mask_comment_opinions = mask_comment_opinions[:, t]
                t_actual_reply_comment = actual_reply_comment[:, t]
                t_actual_comment = actual_comment[:, t]

                reply_comment, user_state = self.choose_reply_comment(t, user_state, t_padded_comments_opinions, t_mask_comment_opinions, actual_reply_comment=t_actual_reply_comment)
                our_comment = self.write_comment(t, user_state, actual_comment=t_actual_comment)
        return reply_comment, our_comment, user_state

    def reply_and_write_comment_guide(self, padded_comments_opinions, mask_comment_opinions, actual_reply_comment=None, actual_comment=None):
        self.choose_reply_comment_guide(padded_comments_opinions, mask_comment_opinions, actual_reply_comment)

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

