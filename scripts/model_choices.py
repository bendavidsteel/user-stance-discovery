import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

def degroot(user_state, new_content, attention):
    return user_state + torch.dot(new_content, attention)

def friedkin_johnsen(original_user_state, content, stubbornness):
    return ((1 - stubbornness) * original_user_state) + (stubbornness * torch.sum(content))

def bounded_confidence(user_state, new_content, confidence_interval):
    diff = user_state - new_content
    mask = torch.norm(torch.abs(diff), dim=1) > confidence_interval
    return user_state + ((1 / torch.sum(mask)) * torch.sum(mask * diff))

def stochastic_bounded_confidence(user_state, new_content, exponent):
    # compares prob of interaction between a number of people
    numerator = torch.pow(torch.abs(user_state - new_content), -1 * exponent)
    return numerator / torch.sum(numerator)

def recommend_videos(available_videos, user_history):
    return torch.full(len(available_videos), 1 / len(available_videos))

def state_update(user_state, new_content, attention):
    return degroot(user_state, new_content, attention)

def get_video_opinions(video):
    return video.topics

def get_comments_opinions(comments):
    return [comment.topic for comment in comments]

def do_choose_video(recommended_video, user_state, diff_weight):
    # TODO this doesn't work for only 1 video, must have multiple interactions
    return stochastic_bounded_confidence(user_state, recommended_video, diff_weight)

def choose_comment(comments, user_state, diff_weight):
    return stochastic_bounded_confidence(user_state, comments, diff_weight)

def write_comment_opinions(user_state):
    return user_state

def model(available_videos, user_history, user_state=None, video=None, chosen_video=None, replied_comment=None, comment=None):

    num_data = len(user_history)
    with pyro.plate("actions", num_data):
        if not user_state:
            user_state = pyro.sample("initial_user_state", dist.MultivariateNormal())

        # tiktok recommends a video
        video_probs = recommend_videos(available_videos, user_history)
        recommended_video = pyro.sample("recommended_video", dist.Categorical(probs=video_probs), obs=video)

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
        choose_video_probs = do_choose_video(video_opinions, user_state, video_diff_weight)
        choose_video = pyro.sample("choose_video", dist.Bernoulli(choose_video_probs), obs=chosen_video)

        if choose_video:
            # comments are then shown on video and causes state update
            video_comments = available_videos[recommended_video]
            comments_opinions = get_comments_opinions(video_comments)
            user_state = state_update(user_state, comments_opinions, comment_attention)

            # select a comment to reply to
            # TODO maybe pick different prior
            comment_diff_weight = pyro.sample("comment_diff_weight", dist.Normal(0, 1))
            comment_probs = choose_comment(comments_opinions, user_state, comment_diff_weight)
            chosen_comment = pyro.sample("choose_comment", dist.Categorical(probs=comment_probs), obs=replied_comment)

            # write a comment
            comment_opinions = write_comment_opinions(user_state)
            our_comment = pyro.sample("our_comment", dist.Normal(loc=comment_opinions, scale=1), obs=comment)


def main():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    adam = pyro.optim.Adam({"lr": 0.02})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    losses = []
    for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
        loss = svi.step(is_cont_africa, ruggedness, log_gdp)
        losses.append(loss)
        if step % 100 == 0:
            logging.info("Elbo loss: {}".format(loss))

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")

if __name__ == '__main__':
    main()