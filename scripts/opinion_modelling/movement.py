import collections
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM

import estimate, opinion_datasets

def plot_movement(dataset, root_dir_path):
    num_people = dataset.num_people

    # opinion_timelines = np.full((dataset.max_time_step, num_people, dataset.num_opinions), np.nan)
    # for i in tqdm.tqdm(range(dataset.max_time_step)):
    #     opinion_mean, opinion_variance, opinion_timeline = dataset[i]
    #     opinion_timelines[i] = opinion_mean
    min_time_step = dataset.min_time_step.to_pydatetime()
    max_time_step = dataset.max_time_step.to_pydatetime()
    # get all months between min and max time step
    starts = []
    ends = []
    start_year = min_time_step.year
    start_month = min_time_step.month
    start = datetime.datetime(start_year, start_month, 1)
    while start < max_time_step:
        if start.month == 12:
            end = datetime.datetime(start.year + 1, 1, 1)
        else:
            end = datetime.datetime(start.year, start.month + 1, 1)
        starts.append(start)
        ends.append(end)
        start = end

    opinion_timelines, opinion_timelines_var, users = dataset.get_data(starts, ends)

    top_num = 3

    # get max movement for each stance
    max_opinion = np.nanmax(opinion_timelines, axis=0)
    min_opinion = np.nanmin(opinion_timelines, axis=0)
    max_opinion_movement = max_opinion - min_opinion

    # plot top 5 movers for each stance
    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), figsize=(10, 2.5*len(dataset.stance_columns)))
    for i in range(len(dataset.stance_columns)):
        max_stance_movement = max_opinion_movement[:, i]
        non_nans = ~np.isnan(max_stance_movement)
        max_stance_movement = max_stance_movement[non_nans]
        stance_timelines = opinion_timelines[:, non_nans, i]
        stance_timelines_var = opinion_timelines_var[:, non_nans, i]
        sorted_stance_movement = np.sort(max_stance_movement)
        sorted_stance_movers = np.argsort(max_stance_movement)
        top_stance_movement = sorted_stance_movement[-top_num:]
        top_stance_movers = sorted_stance_movers[-top_num:]
        for j in range(top_num):
            axes[i].errorbar(ends, stance_timelines[:, top_stance_movers[-j-1]], yerr=stance_timelines_var[:, top_stance_movers[-j-1]])
        axes[i].set_title(dataset.stance_columns[i].replace('stance_', '').replace('_', ' ').title())
    fig.savefig(os.path.join(root_dir_path, "figs", "top_stance_movers.png"))
    

    # get max movement for all stances
    max_movement = np.zeros((opinion_timelines.shape[1],))
    max_movement_indices = np.zeros((opinion_timelines.shape[1], 2))
    for i in range(opinion_timelines.shape[0]):
        for j in range(i+1, opinion_timelines.shape[0]):
            movement = np.sqrt(np.sum((opinion_timelines[i] - opinion_timelines[j])**2, axis=1))
            max_movement = np.select([movement > max_movement], [movement], default=max_movement)
            max_movement_indices = np.select(
                [np.broadcast_to((movement > max_movement), (2, max_movement.shape[0])).T], 
                [np.broadcast_to(np.array([i, j]), max_movement_indices.shape)], 
                default=max_movement_indices
            )

    sorted_max_movement = np.sort(max_movement)
    sorted_max_movers = np.argsort(max_movement)
    top_movement = sorted_max_movement[-top_num:]
    top_movers = sorted_max_movers[-top_num:]

    # plot top 5 movers
    # map users to colors
    user_to_color = {i: plt.cm.tab10(i) for i in range(5)}
    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), figsize=(10, 2.5*len(dataset.stance_columns)))
    for i in range(len(dataset.stance_columns)):
        for j in range(top_num):
            axes[i].errorbar(ends, opinion_timelines[:, top_movers[-j-1], i], yerr=opinion_timelines_var[:, top_movers[-j-1], i], color=user_to_color[j])
        axes[i].set_title(dataset.stance_columns[i].replace('stance_', '').replace('_', ' ').title())
    fig.savefig(os.path.join(root_dir_path, "figs", "top_movers.png"))


    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), figsize=(10, 2.5*len(dataset.stance_columns)))
    for i in range(len(dataset.stance_columns)):
        axes[i].plot(ends, opinion_timelines[:, :, i])
        axes[i].set_title(dataset.stance_columns[i].replace('stance_', '').replace('_', ' ').title())

    fig.savefig(os.path.join(root_dir_path, "figs", "opinion_timelines.png"))

def fit_hmm(dataset):
    dataset.aggregation = None
    opinion_sequences, users, classifier_indices = dataset.get_data(start=dataset.min_time_step, end=dataset.max_time_step)

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)

    for i, stance in enumerate(dataset.stance_columns):
        estimator.set_stance(stance)

        X = []
        classifier_counts = {i: {j: 0 for j in range(len(dataset.all_classifier_profiles[stance]))} for i in range(3)}
        for j in range(opinion_sequences.shape[0]):
            seq = opinion_sequences[j, np.where(~np.isnan(opinion_sequences[j, :, i]))[0], i]
            seq = torch.tensor(seq.astype(int) + 1).reshape((-1, 1))
            if seq.shape[0] > 0:
                X.append(seq)

            for k in range(opinion_sequences.shape[1]):
                if not np.isnan(opinion_sequences[j, k, i]):
                    if opinion_sequences[j, k, i] == -1:
                        classifier_counts[0][classifier_indices[j, k]] += 1
                    elif opinion_sequences[j, k, i] == 0:
                        classifier_counts[1][classifier_indices[j, k]] += 1
                    elif opinion_sequences[j, k, i] == 1:
                        classifier_counts[2][classifier_indices[j, k]] += 1

        # normalize classifier_counts
        for j in range(3):
            j_sum = np.sum(list(classifier_counts[j].values()))
            for k in range(len(dataset.all_classifier_profiles[stance])):
                classifier_counts[j][k] /= j_sum

        # batch sequences by length
        X_dict = collections.defaultdict(list)
        for i, x in enumerate(X):
            n = len(x)
            X_dict[n].append(x)

        keys = sorted(X_dict.keys())
        X = [torch.stack(X_dict[key]) for key in keys]

        # Get prediction probabilities based on the confusion matrix
        # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
        confusion_probs = estimator.predictor_confusion_probs[stance]['predict_probs']
        against_predict_probs = torch.sum(torch.tensor([(confusion_probs[c_idx, 0, :] * classifier_counts[0][c_idx]).tolist() for c_idx in classifier_counts[0]]), axis=0)
        neutral_predict_probs = torch.sum(torch.tensor([(confusion_probs[c_idx, 1, :] * classifier_counts[1][c_idx]).tolist() for c_idx in classifier_counts[1]]), axis=0)
        favor_predict_probs = torch.sum(torch.tensor([(confusion_probs[c_idx, 2, :] * classifier_counts[2][c_idx]).tolist() for c_idx in classifier_counts[2]]), axis=0)

        against = Categorical([against_predict_probs.tolist()])
        neutral = Categorical([neutral_predict_probs.tolist()])
        favor = Categorical([favor_predict_probs.tolist()])

        edges = [[0.9, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]
        starts = [0.8, 0.1, 0.1]

        model = DenseHMM([against, neutral, favor], edges=edges, starts=starts, verbose=False)
        X_ = torch.cat([x.reshape(-1, 1) for x in X], dim=0).unsqueeze(0) # Add this
        model._initialize(X_) # Add this too
        model.fit(X)

        print(f"Stance: {stance}")
        for i, stance_i in enumerate(["against", "neutral", "favor"]):
            for j, stance_j in enumerate(["against", "neutral", "favor"]):
                print(f"{stance_i} -> {stance_j}: {torch.exp(model.edges[i, j]):.3f}")

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    dataset_name = "reddit"
    aggregation = "weighted_exponential_smoothing"

    if dataset_name == "reddit":
        dataset = opinion_datasets.RedditOpinionTimelineDataset(aggregation=aggregation, halflife=100., min_num_per_stance=50)
    elif dataset_name == "generative":
        num_people = 10
        max_time_step = 10
        num_opinions = 3
        dataset = opinion_datasets.GenerativeOpinionTimelineDataset(num_people=num_people, max_time_step=max_time_step, num_opinions=num_opinions)

    do_plot_movement = False
    if do_plot_movement:
        plot_movement(dataset, root_dir_path)

    do_hmm = True
    if do_hmm:
        fit_hmm(dataset)

if __name__ == "__main__":
    main()