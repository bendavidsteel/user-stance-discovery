import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import pandas as pd
import torch
import tqdm

from mining.generative import SocialGenerativeModel, UpdateDist

def main():
    # Fixing random state for reproducibility
    torch.manual_seed(19680801)

    display = True

    num_steps = 100
    num_users = 100
    social_context = SocialGenerativeModel(num_users=num_users)
    if display:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        ud = UpdateDist(social_context, axes[0], axes[1])
        anim = FuncAnimation(fig, ud, frames=num_steps, interval=100, blit=True)
        plt.show()
    else:
        social_context.reset()
        for i in tqdm.tqdm(range(num_steps)):
            social_context.update(i)

if __name__ == '__main__':
    main()