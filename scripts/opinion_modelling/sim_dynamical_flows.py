import datetime

import gpytorch
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from mining.datasets import GenerativeOpinionTimelineDataset
from mining.dynamics import flow_pairplot
from mining.estimate import prep_gp_data, train_gaussian_process, get_gp_means
from mining.generative import SocialGenerativeModel, UpdateDist


def main():
    num_steps = 100
    num_users = 10
    num_opinions = 8
    social_context = SocialGenerativeModel(
        num_users=num_users, 
        sbc_exponent_loc=1.0,
        sbc_exponent_scale=0.1,
        comment_prob=0.2, 
        sus_loc=0.8, 
        sus_scale=0.1,
        seen_att_loc=0.001,
        reply_att_loc=0.1,
        post_att_loc=0.05,
        content_scale=0.1,
        num_opinions=num_opinions,
    )
    interval = 10
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    ud = UpdateDist(social_context, axes[0], axes[1], num_steps=num_steps, interval=interval)
    anim = FuncAnimation(fig, ud, frames=num_steps//interval, interval=10, blit=True)
    anim.save('./figs/flows/gen.mp4')
    
    dataset = GenerativeOpinionTimelineDataset(generative_model=social_context)

    X_norm, y = prep_gp_data(dataset)
    model_list, likelihood_list, losses = train_gaussian_process(X_norm, y)
    timestamps, means = get_gp_means(dataset, model_list, likelihood_list, X_norm, y)

    min_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
    max_time = datetime.datetime(2023, 12, 31, 23, 59, 59)

    unix_timestamps = min_time.timestamp() + (max_time.timestamp() - min_time.timestamp()) * timestamps / np.max(timestamps)
    datetimes = np.array([datetime.datetime.fromtimestamp(t) for t in unix_timestamps])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")  # supress output text
    fig.savefig(f'./figs/flows/losses.png')

    figs, axes = flow_pairplot(datetimes, means, dataset.stance_columns, do_pairplot=False)

    for i in range(num_opinions):
        for j in range(num_opinions):
            figs[i,j].savefig(f'./figs/flows/{i}_{j}_flows.png')

            
if __name__ == '__main__':
    main()

