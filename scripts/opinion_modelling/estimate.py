import os

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine

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
                        predict_probs[predictor_id, true_idx, :] = torch.tensor([1/3, 1/3, 1/3])

                for predicted_idx, predicted_cat in enumerate(["predicted_against", "predicted_neutral", "predicted_favor"]):
                    try:
                        true_probs[predictor_id, predicted_idx, :] = get_true_probs(predicted_cat, confusion_profile)
                    except ZeroDivisionError:
                        true_probs[predictor_id, predicted_idx, :] = torch.tensor([1/3, 1/3, 1/3])

            self.predictor_confusion_probs[stance] = {
                'predict_probs': predict_probs,
                'true_probs': true_probs,
            }

    def set_stance(self, stance):
        self.stance = stance

    def model(self, opinion_sequences, predictor_ids, mask, prior=False):
        
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            if prior:
                user_stance_loc = pyro.param("user_stance_loc", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.interval(-1, 1))
                user_stance_loc_var = pyro.param("user_stance_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                user_stance = pyro.sample("user_stance", dist.Normal(user_stance_loc, user_stance_loc_var))
                # user_stance_var = pyro.param("user_stance_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                user_stance_var = torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1))
                
            else:
                user_stance_var = pyro.param("user_stance_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                # # sample stance from the uniform prior
                user_stance = pyro.param("user_stance", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.interval(-1, 1))

            # loop over the observed data
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # User creates comments with latent stance
                    # Standard deviation should be half the distance between categories
                    # comment_var = (1/2) ** 2
                    
                    if prior:
                        comment_stance_cats = torch.zeros_like(opinion_sequences, dtype=torch.int)
                        comment_stance_cats[user_stance.repeat(1, comment_stance_cats.shape[1]) > 1/2] = 2
                        comment_stance_cats[user_stance.repeat(1, comment_stance_cats.shape[1]) < -1/2] = 0
                        comment_stance_cats[(user_stance.repeat(1, comment_stance_cats.shape[1]) >= -1/2) & (user_stance.repeat(1, comment_stance_cats.shape[1]) <= 1/2)] = 1
                    else:
                        comment_stances = pyro.sample("latent_comment_stance", dist.Normal(user_stance, user_stance_var).expand(opinion_sequences.shape))

                        # Quantize comment stance into 3 categories
                        comment_stance_cats = torch.zeros_like(comment_stances, dtype=torch.int)
                        comment_stance_cats[comment_stances > 1/2] = 2
                        comment_stance_cats[comment_stances < -1/2] = 0
                        comment_stance_cats[(comment_stances >= -1/2) & (comment_stances <= 1/2)] = 1

                    # Get prediction probabilities based on the confusion matrix
                    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                    predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

                    pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=opinion_sequences)

    def guide(self, opinion_sequences, predictor_ids, mask, prior=False):
        # comment_stance_var_loc = pyro.param("comment_stance_var_loc", torch.tensor(0.1))
        # comment_stance_var_scale = pyro.param("comment_stance_var_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
        # comment_stance_var = pyro.sample("comment_stance_var", dist.LogNormal(comment_stance_var_loc, comment_stance_var_scale), infer={'is_auxiliary': True})
        # # sample stance from the uniform prior
        
        # loop over the observed data
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            if prior:
                user_stance_loc = pyro.param("user_stance_loc_q", torch.tensor(0.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.interval(-1, 1))
                user_stance_loc_var = pyro.param("user_stance_var_q", torch.tensor(0.001).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
                user_stance = pyro.sample("user_stance", dist.Normal(user_stance_loc, user_stance_loc_var))
            comment_var = pyro.param("comment_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):

                    # Get true probabilities from the confusion matrix
                    true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, opinion_sequences, :]

                    # Sample latent comment stance categories
                    comment_stance_cats = pyro.sample("latent_comment_stance_category", dist.Categorical(probs=true_probs), infer={'is_auxiliary': True})

                    # Determine latent locations based on categories
                    latent_locs = torch.zeros_like(comment_stance_cats, dtype=torch.float)
                    latent_locs[comment_stance_cats == 1] = 0
                    latent_locs[comment_stance_cats == 2] = 1
                    latent_locs[comment_stance_cats == 0] = -1

                    # Sample latent comment stances
                    # comment_stances = pyro.sample("latent_comment_stance", dist.Normal(latent_locs, comment_var))
                    if not prior:
                        comment_stances = pyro.sample("latent_comment_stance", dist.Delta(latent_locs))

    def beta_model(self, opinion_sequences, predictor_ids, mask, prior=False):
        if prior:
            alpha = pyro.sample("alpha", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
            beta = pyro.sample("beta", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
        else:
            alpha = pyro.param("alpha", torch.tensor(10.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)
            beta = pyro.param("beta", torch.tensor(10.0).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.positive)

        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            # loop over the observed data
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # User creates comments with latent stance
                    # Standard deviation should be half the distance between categories
                    # comment_var = (1/2) ** 2
                    comment_stances = pyro.sample("latent_comment_stance", dist.Beta(alpha, beta).expand(opinion_sequences.shape))

                    # Quantize comment stance into 3 categories
                    comment_stance_cats = torch.zeros_like(comment_stances, dtype=torch.int)
                    comment_stance_cats[comment_stances > 2/3] = 2
                    comment_stance_cats[comment_stances < 1/3] = 0
                    comment_stance_cats[(comment_stances >= 1/3) & (comment_stances <= 2/3)] = 1

                    # Get prediction probabilities based on the confusion matrix
                    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                    predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

                    pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=opinion_sequences)

    def beta_guide(self, opinion_sequences, predictor_ids, mask, prior=False):
        # comment_stance_var_loc = pyro.param("comment_stance_var_loc", torch.tensor(0.1))
        # comment_stance_var_scale = pyro.param("comment_stance_var_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
        # comment_stance_var = pyro.sample("comment_stance_var", dist.LogNormal(comment_stance_var_loc, comment_stance_var_scale), infer={'is_auxiliary': True})
        # # sample stance from the uniform prior
        # user_stance_loc = pyro.param("user_stance_loc", torch.tensor(0.0), constraint=dist.constraints.interval(-1, 1))
        # user_stance = pyro.sample("user_stance", dist.Normal(user_stance_loc, user_stance_var))
        # loop over the observed data
        comment_var = pyro.param("comment_var", torch.tensor(0.1).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.interval(0, 0.5**2))

        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            # Standard deviation should be half the distance between categories
            with pyro.plate("observed_data", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # Get true probabilities from the confusion matrix
                    true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, opinion_sequences, :]

                    # Sample latent comment stance categories
                    comment_stance_cats = pyro.sample("latent_comment_stance_category", dist.Categorical(probs=true_probs), infer={'is_auxiliary': True})

                    # Determine latent locations based on categories
                    latent_locs = torch.zeros_like(comment_stance_cats, dtype=torch.float)
                    latent_locs[comment_stance_cats == 1] = 0.5
                    latent_locs[comment_stance_cats == 2] = 1
                    latent_locs[comment_stance_cats == 0] = 0

                    # map to support of beta
                    latent_locs = torch.clamp(latent_locs, 1e-6, 1 - 1e-6)
                    # comment_var = torch.min(comment_var, latent_locs * (1 - latent_locs))
                    # alpha = torch.maximum((((1 - latent_locs) / comment_var) - (1 / latent_locs)) * (latent_locs ** 2), torch.tensor(1e-6))
                    # beta = torch.maximum(alpha * ((1 / latent_locs) - 1), torch.tensor(1e-6))

                    # Sample latent comment stances
                    comment_stances = pyro.sample("latent_comment_stance", dist.Delta(latent_locs))
                    # comment_stances = pyro.sample("latent_comment_stance", dist.Beta(alpha, beta))


    def categorical_model(self, opinion_sequences, predictor_ids, mask):
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            user_stance = pyro.param("user_stance", torch.tensor([1/3, 1/3, 1/3]).unsqueeze(0).tile((opinion_sequences.shape[0],1)), constraint=dist.constraints.simplex)

            # loop over the observed data
            with pyro.plate("observed_data_sequence", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):
                    # User creates comments with latent stance
                    # Standard deviation should be half the distance between categories
                    # comment_var = (1/2) ** 2
                    comment_stance_cats = pyro.sample(
                        "latent_comment_stance", 
                        dist.Categorical(probs=user_stance.unsqueeze(1)).expand(opinion_sequences.shape)
                    )

                    # Get prediction probabilities based on the confusion matrix
                    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
                    predict_probs = self.predictor_confusion_probs[self.stance]['predict_probs'][predictor_ids, comment_stance_cats, :]

                    pyro.sample("predicted_comment_stance", dist.Categorical(probs=predict_probs), obs=opinion_sequences)


    def categorical_guide(self, opinion_sequences, predictor_ids, mask):
        with pyro.plate("batched_data", opinion_sequences.shape[0], dim=-2):
            # loop over the observed data
            with pyro.plate("observed_data_sequence", opinion_sequences.shape[1], dim=-1):
                with poutine.mask(mask=mask):

                    # Get true probabilities from the confusion matrix
                    true_probs = self.predictor_confusion_probs[self.stance]['true_probs'][predictor_ids, opinion_sequences, :]

                    # Sample latent comment stance categories
                    comment_stance_cats = pyro.sample("latent_comment_stance", dist.Categorical(probs=true_probs))


def get_inferred_categorical(dataset, opinion_sequences, all_classifier_indices):
    estimator = StanceEstimation(dataset.all_classifier_profiles)

    user_stances = np.zeros((opinion_sequences.shape[0], len(dataset.stance_columns), 3))

    # setup the optimizer
    adam_params = {"lr": 0.001}
    optimizer = Adam(adam_params)
    # for user_idx in range(opinion_sequences.shape[0]):
    for stance_idx, stance_column in enumerate(dataset.stance_columns):

        op_seqs = []
        lengths = []
        all_predictor_ids = []
        for user_idx in range(opinion_sequences.shape[0]):
            seq = []
            length = 0
            predictor_ids = []
            for i in range(opinion_sequences.shape[1]):
                if not np.isnan(opinion_sequences[user_idx, i, stance_idx]):
                    seq.append(opinion_sequences[user_idx, i, stance_idx])
                    length += 1
                    predictor_ids.append(all_classifier_indices[user_idx, i].astype(int))

            op_seqs.append(np.array(seq))
            lengths.append(length)
            all_predictor_ids.append(np.array(predictor_ids))

        max_length = max(lengths)

        stance_opinion_sequences = np.zeros((opinion_sequences.shape[0], max_length))
        classifier_indices = np.zeros((opinion_sequences.shape[0], max_length))
        mask = np.zeros((opinion_sequences.shape[0], max_length))
        for i in range(opinion_sequences.shape[0]):
            stance_opinion_sequences[i, :lengths[i]] = op_seqs[i]
            classifier_indices[i, :lengths[i]] = all_predictor_ids[i]
            mask[i, :lengths[i]] = 1

        if stance_column not in estimator.predictor_confusion_probs or len(estimator.predictor_confusion_probs[stance_column]) == 0:
            continue
        estimator.set_stance(stance_column)
        stance_opinion_sequences = torch.tensor(stance_opinion_sequences).int() + 1
        classifier_indices = torch.tensor(classifier_indices).int()
        mask = torch.tensor(mask).bool()

        pyro.clear_param_store()
        # do gradient steps
        svi = SVI(estimator.categorical_model, estimator.categorical_guide, optimizer, loss=Trace_ELBO())
        n_steps = 1000
        for step in range(n_steps):
            svi.step(stance_opinion_sequences, classifier_indices, mask)

        # grab the learned variational parameters
        user_stance = pyro.param("user_stance").detach().numpy()
        user_stances[:, stance_idx] = user_stance

    return user_stances

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