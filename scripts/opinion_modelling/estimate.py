import os

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import opinion_datasets

class StanceEstimation:
    def __init__(self, all_classifier_profiles):

        def get_predict_probs(true_cat, confusion_profile):
            predict_sum = sum(v for k, v in confusion_profile[true_cat].items())
            predict_probs = torch.tensor([
                confusion_profile[true_cat]["predicted_against"] / predict_sum,
                confusion_profile[true_cat]["predicted_neutral"] / predict_sum,
                confusion_profile[true_cat]["predicted_favor"] / predict_sum,
            ])
            return predict_probs
        
        def get_true_probs(predicted_cat, confusion_profile):
            true_sum = sum(v[predicted_cat] for k, v in confusion_profile.items())
            true_probs = torch.tensor([
                confusion_profile["true_against"][predicted_cat] / true_sum,
                confusion_profile["true_neutral"][predicted_cat] / true_sum,
                confusion_profile["true_favor"][predicted_cat] / true_sum,
            ])
            return true_probs

        self.predictor_confusion_probs = {}
        for stance in all_classifier_profiles:
            
            predict_probs = torch.zeros(len(all_classifier_profiles[stance]), 3, 3)
            true_probs = torch.zeros(len(all_classifier_profiles[stance]), 3, 3)

            assert len(all_classifier_profiles[stance]) == max(all_classifier_profiles[stance].keys()) + 1
            for predictor_id in all_classifier_profiles[stance]:
                classifier_profile = all_classifier_profiles[stance][predictor_id]

                try:
                    confusion_profile = {
                        "true_favor": classifier_profile["true_favor"],
                        "true_against": classifier_profile["true_against"],
                        "true_neutral": classifier_profile["true_neutral"],
                    }
                except KeyError:
                    continue

                for true_idx, true_cat in enumerate(["true_against", "true_neutral", "true_favor"]):
                    try:
                        predict_probs[predictor_id, true_idx, :] = get_predict_probs(true_cat, confusion_profile)
                    except ZeroDivisionError:
                        continue

                for predicted_idx, predicted_cat in enumerate(["predicted_against", "predicted_neutral", "predicted_favor"]):
                    try:
                        true_probs[predictor_id, predicted_idx, :] = get_true_probs(predicted_cat, confusion_profile)
                    except ZeroDivisionError:
                        continue

            self.predictor_confusion_probs[stance] = {
                'predict_probs': predict_probs,
                'true_probs': true_probs,
            }

    def set_stance(self, stance):
        self.stance = stance

    def model(self, data):
        user_stance_var = pyro.param("user_stance_var", torch.tensor(0.1), constraint=dist.constraints.positive)
        # # sample stance from the uniform prior
        user_stance = pyro.param("user_stance", torch.tensor(0.0), constraint=dist.constraints.interval(-1, 1))
        # user_stance = pyro.sample("user_stance", dist.Normal(user_stance_loc, user_stance_var))

        # loop over the observed data
        with pyro.plate("observed_data", len(data)):
            predictor_ids = torch.tensor([d['predictor_id'] for d in data])
            
            # User creates comments with latent stance
            # Standard deviation should be half the distance between categories
            # comment_var = (1/2) ** 2
            comment_stances = pyro.sample("latent_comment_stance", dist.Normal(user_stance, user_stance_var).expand([len(data)]))

            # Quantize comment stance into 3 categories
            comment_stance_cats = torch.zeros(len(comment_stances), dtype=torch.int)
            comment_stance_cats[comment_stances > 1/2] = 2
            comment_stance_cats[comment_stances < -1/2] = 0
            comment_stance_cats[(comment_stances >= -1/2) & (comment_stances <= 1/2)] = 1

            # Get prediction probabilities based on the confusion matrix
            # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
            predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

            # Map predicted stances
            obs_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_predicted_stances = torch.tensor([obs_mapping[d['predicted_stance']] for d in data])

            pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=mapped_predicted_stances)

    def guide(self, data):
        # comment_stance_var_loc = pyro.param("comment_stance_var_loc", torch.tensor(0.1))
        # comment_stance_var_scale = pyro.param("comment_stance_var_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
        # comment_stance_var = pyro.sample("comment_stance_var", dist.LogNormal(comment_stance_var_loc, comment_stance_var_scale), infer={'is_auxiliary': True})
        # # sample stance from the uniform prior
        # user_stance_loc = pyro.param("user_stance_loc", torch.tensor(0.0), constraint=dist.constraints.interval(-1, 1))
        # user_stance = pyro.sample("user_stance", dist.Normal(user_stance_loc, user_stance_var))
        # loop over the observed data
        with pyro.plate("observed_data", len(data)):
            predictor_ids = torch.tensor([d['predictor_id'] for d in data])

            # Map predicted stances
            obs_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_predicted_stances = torch.tensor([obs_mapping[d['predicted_stance']] for d in data])

            # Get true probabilities from the confusion matrix
            true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, mapped_predicted_stances, :]

            # Sample latent comment stance categories
            comment_stance_cats = pyro.sample("latent_comment_stance_category", dist.Categorical(probs=true_probs), infer={'is_auxiliary': True})

            # Determine latent locations based on categories
            latent_locs = torch.zeros_like(comment_stance_cats, dtype=torch.float)
            latent_locs[comment_stance_cats == 1] = 0
            latent_locs[comment_stance_cats == 2] = 1
            latent_locs[comment_stance_cats == 0] = -1

            # Standard deviation should be half the distance between categories
            comment_var = (1/2) ** 2

            # Sample latent comment stances
            comment_stances = pyro.sample("latent_comment_stance", dist.Normal(latent_locs, comment_var))

    def categorical_model(self, data):
        user_stance = pyro.param("user_stance", torch.tensor([1/3, 1/3, 1/3]), constraint=dist.constraints.simplex)

        # loop over the observed data
        with pyro.plate("observed_data", len(data)):
            predictor_ids = torch.tensor([d['predictor_id'] for d in data])
            
            # User creates comments with latent stance
            # Standard deviation should be half the distance between categories
            # comment_var = (1/2) ** 2
            comment_stance_cats = pyro.sample("latent_comment_stance", dist.Categorical(probs=user_stance).expand([len(data)]))

            # Get prediction probabilities based on the confusion matrix
            # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
            predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

            # Map predicted stances
            obs_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_predicted_stances = torch.tensor([obs_mapping[d['predicted_stance']] for d in data])

            pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=mapped_predicted_stances)


    def categorical_guide(self, data):
        # loop over the observed data
        with pyro.plate("observed_data", len(data)):
            predictor_ids = torch.tensor([d['predictor_id'] for d in data])

            # Map predicted stances
            obs_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_predicted_stances = torch.tensor([obs_mapping[d['predicted_stance']] for d in data])

            # Get true probabilities from the confusion matrix
            true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, mapped_predicted_stances, :]

            # Sample latent comment stance categories
            comment_stance_cats = pyro.sample("latent_comment_stance", dist.Categorical(probs=true_probs))


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