import os

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine

from mining.estimate import StanceEstimation

import opinion_datasets


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    dataset_name = "generative"

    if dataset_name == "reddit":
        dataset = opinion_datasets.RedditOpinionTimelineDataset()
    elif dataset_name == "generative":
        num_people = 10
        max_time_step = 10
        num_opinions = 3
        dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(num_people=num_people, max_time_step=max_time_step, num_opinions=num_opinions)

    max_time_step = dataset.max_time_step
    opinion_sequences, mask_sequence, users, classifier_indices = dataset[max_time_step]

    estimator = StanceEstimation(dataset.all_classifier_profiles)

    # setup the optimizer
    adam_params = {"lr": 0.001}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    # autoguide_map = pyro.infer.autoguide.AutoDelta(estimator.model)

    for user_idx in range(opinion_sequences.shape[0]):
        for stance_idx, stance_column in enumerate(dataset.stance_columns):

            data = []
            for i in range(opinion_sequences.shape[1]):
                if mask_sequence[user_idx, i] == 1 and not np.isnan(opinion_sequences[user_idx, i, stance_idx]):
                    data.append({
                        'predicted_stance': opinion_sequences[user_idx, i, stance_idx],
                        'predictor_id': classifier_indices[user_idx, i]
                    })

            if len(data) == 0:
                continue

            stance = stance_column.replace('stance_', '')
            if stance not in estimator.predictor_confusion_probs or len(estimator.predictor_confusion_probs[stance]) == 0:
                continue
            estimator.set_stance(stance)

            # do gradient steps
            svi = SVI(estimator.model, estimator.guide, optimizer, loss=Trace_ELBO())
            n_steps = 1000
            for step in range(n_steps):
                svi.step(data)

            # grab the learned variational parameters
            user_stance = pyro.param("user_stance").item()
            user_stance_variance = pyro.param("user_stance_var").item()
            

            # compare to user stance and variance suggested by data
            data_user_stance = np.mean([d['predicted_stance'] for d in data])
            data_user_stance_variance = np.var([d['predicted_stance'] for d in data])

            print(f"User stance: {user_stance}, {user_stance_variance}, from data: {data_user_stance}, {data_user_stance_variance}")

if __name__ == '__main__':
    main()