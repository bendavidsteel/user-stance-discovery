import os

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import tqdm

import opinion_datasets, estimate


def get_error(user_stance, user_stance_variance, num_data_points, prediction_precision=1.0):
    dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, prediction_precision=prediction_precision)
    max_time_step = dataset.max_time_step
    opinion_sequences, mask_sequence, users, classifier_indices = dataset[max_time_step]
    data = []
    for i in range(opinion_sequences.shape[1]):
        if mask_sequence[0, i] == 1 and not np.isnan(opinion_sequences[0, i, 0]):
            data.append({
                'predicted_stance': opinion_sequences[0, i, 0],
                'predictor_id': classifier_indices[0, i]
            })

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])

    method = "direct"

    if method == "SVI":
        # guide = pyro.infer.autoguide.AutoDelta(estimator.model)
        guide = estimator.guide

        # setup the optimizer
        num_steps = 500
        initial_lr = 0.01
        gamma = 0.5  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / num_steps)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
        svi = SVI(estimator.model, guide, optim, loss=Trace_ELBO())
        losses = []
        predicted_stances = []
        predicted_variances = []
        for step in range(num_steps):
            loss = svi.step(data)
            losses.append(loss)
            predicted_stances.append(pyro.param("user_stance").item())
            predicted_variances.append(pyro.param("user_stance_var").item())

        # grab the learned variational parameters
        predicted_stance = pyro.param("user_stance").item()
        variance = pyro.param("user_stance_var").item()
    
    elif method == "MCMC":
        nuts_kernel = pyro.infer.mcmc.NUTS(estimator.model)
        mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=100, warmup_steps=50)
        mcmc.run(data)
        losses = []
        predicted_stances = []
        predicted_variances = []
        # for sample in mcmc.get_samples():
        #     losses.append(0)
        #     predicted_stances.append(sample["user_stance_loc"].item())
        #     predicted_variances.append(sample["user_stance_var"].item())

        samples = mcmc.get_samples()
        predicted_stance = samples["user_stance_loc"]
        variance = samples["user_stance_var"]
    elif method == "direct":
        losses = []
        predicted_stances = []
        predicted_variances = []
        predicted_stance, variance = estimator.calculate_statistics(data)


    return predicted_stance, variance, losses, predicted_stances, predicted_variances

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")
    figs_dir_path = os.path.join(root_dir_path, "figs")
    
    # plot loss
    user_stance = 1.0
    user_stance_variance = 0.1
    num_data_points = 100
    prediction_precision = 0.6
    _, _, losses, predicted_stances, predicted_variances = get_error(user_stance, user_stance_variance, num_data_points, prediction_precision=prediction_precision)
    fig, ax = plt.subplots()
    ax.plot(losses, color="red")
    ax1 = ax.twinx()
    ax1.errorbar(np.arange(len(predicted_stances)), predicted_stances, yerr=predicted_variances, label="Predicted stance")
    ax1.errorbar(np.arange(len(predicted_stances)), [user_stance] * len(predicted_stances), yerr=[user_stance_variance] * len(predicted_stances), label="True stance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax1.set_ylabel("Predicted stance")
    ax1.legend()
    fig.savefig(os.path.join(figs_dir_path, "test_estimate_loss.png"))

    if True:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 100
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _ = get_error(user_stance, user_stance_variance, num_data_points)
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")

        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_user_stance.png"))

        # test sensitivity to user stance with high variance
        user_stance_variance = 1
        num_data_points = 100
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _ = get_error(user_stance, user_stance_variance, num_data_points)
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")
        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_user_stance_high_variance.png"))

    if True:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 5
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _ = get_error(user_stance, user_stance_variance, num_data_points)
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")

        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_num_data_points.png"))

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        prediction_precision = 0.6
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _ = get_error(user_stance, user_stance_variance, num_data_points, prediction_precision=prediction_precision)
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")
        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_prediction_precision.png"))



if __name__ == '__main__':
    main()