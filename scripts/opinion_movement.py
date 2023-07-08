import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

class UpdateDist:
    def __init__(self, ax):
        num_users = 100
        self.users = torch.zeros((num_users, 2))
        self.reset_users()

        self.line, = ax.scatter(self.users[:, 0], self.users[:, 1])
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True)

    def reset_users(self):
        for i in range(self.users.shape[0]):
            self.users[i] = torch.random.uniform(-1, 1, 2)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.reset_users()
            self.line.set_data(self.users[:,0], self.users[:,1])
            return self.line,

        
        self.line.set_data(self.users[:,0], self.users[:,1])
        return self.line,

def main():
    # Fixing random state for reproducibility
    torch.random.seed(19680801)


    fig, ax = plt.subplots()
    ud = UpdateDist(ax, prob=0.7)
    anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
    plt.show()

if __name__ == '__main__':
    main()