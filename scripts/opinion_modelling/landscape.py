
import numpy as np
from sklearn.neighbors import KernelDensity

import opinion_datasets

def main():
    max_time_step = 1000
    dataset = opinion_datasets.GenerativeOpinionTimelineDataset(num_people=100, max_time_step=max_time_step)

    all_opinions = []
    for i in range(max_time_step):
        opinions = dataset[i]
        all_opinions.extend(opinions)

    all_opinions = np.array(all_opinions)

    # create points for each point in the mesh of the multi dimensional opinion space
    opinion_points = np.meshgrid(*[np.linspace(-1, 1, 100) for i in range(all_opinions.shape[1])])
    opinion_points = np.array([opinion_point.ravel() for opinion_point in opinion_points]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(all_opinions)
    log_dens = kde.score_samples(opinion_points)
    dens = np.exp(log_dens)

    # get maximas through the  gradient of the density

if __name__ == '__main__':
    main()