import os

import matplotlib.pyplot as plt
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
import tqdm

import opinion_datasets, estimate


def get_error(user_stance, user_stance_variance, num_data_points, predict_profile='perfect', method='SVI'):
    dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type=predict_profile)
    max_time_step = dataset.max_time_step
    opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=max_time_step)
    data = []
    for i in range(opinion_sequences.shape[1]):
        if not np.isnan(opinion_sequences[0, i, 0]):
            data.append({
                'predicted_stance': opinion_sequences[0, i, 0],
                'predictor_id': classifier_indices[0, i]
            })

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])

    pyro.get_param_store().clear()
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
    
    elif method == "SVI_categorical":
        guide = estimator.categorical_guide

        # setup the optimizer
        num_steps = 500
        initial_lr = 0.01
        gamma = 0.5  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / num_steps)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
        svi = SVI(estimator.categorical_model, guide, optim, loss=Trace_ELBO())
        losses = []
        predicted_stances = []
        predicted_variances = []
        for step in range(num_steps):
            loss = svi.step(data)
            losses.append(loss)
            predicted_stances.append(pyro.param("user_stance").detach().numpy())

        # grab the learned variational parameters
        predicted_stance = pyro.param("user_stance").detach().numpy()
        variance = None

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


        predicted_stances = []
        predicted_variances = []
    
    dataset.aggregation = "weighted_mean"
    freq_means, freq_variances, _ = dataset.get_data(start=0, end=max_time_step)
    freq_mean, freq_variance = freq_means[0,0], freq_variances[0,0]
    dataset.aggregation = None
    precision = dataset.all_classifier_profiles['stance_0'][0]['macro']['precision']
    recall = dataset.all_classifier_profiles['stance_0'][0]['macro']['recall']

    return predicted_stance, variance, losses, predicted_stances, predicted_variances, freq_mean, freq_variance, precision, recall

def plot_experiment(user_stance_variance, num_data_points, predict_profile, method, fig_path, fig_title):
    user_stances = []
    predicted_stances = []
    predicted_variances = []
    freq_stances = []
    freq_variances = []
    for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
        predicted_stance, predicted_variance, _, _, _, freq_mean, freq_var, precision, recall = get_error(user_stance, user_stance_variance, num_data_points, predict_profile=predict_profile, method=method)
        user_stances.append(user_stance)
        predicted_stances.append(predicted_stance)
        predicted_variances.append(predicted_variance)
        freq_stances.append(freq_mean)
        freq_variances.append(freq_var)

    fig, ax = plt.subplots()
    user_stances = np.array(user_stances)
    ax.plot(user_stances, user_stances, label="True stance")
    if method == "SVI":
        ax.errorbar(user_stances + 0.005, predicted_stances, yerr=predicted_variances, label="Predicted stance")
        ax.errorbar(user_stances - 0.005, freq_stances, yerr=freq_variances, label="Frequentist stance")
    elif method == "SVI_categorical":
        ax.matshow(np.array(predicted_stances).T, extent=[-1, 1, 1, -1], aspect='auto', vmin=0, vmax=1, alpha=0.5)
        ax.errorbar(user_stances, freq_stances, yerr=freq_variances, label="Frequentist stance")
    
    ax.set_title(fig_title.format(user_stance_variance=user_stance_variance, num_data_points=num_data_points, precision=precision, recall=recall))
    ax.set_xlabel("True stance")
    ax.set_ylabel("Predicted stance")
    ax.legend()
    fig.savefig(fig_path)

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")
    figs_dir_path = os.path.join(root_dir_path, "figs")
    
    method = 'SVI'

    # plot loss
    user_stance = -1.0
    user_stance_variance = 0.5
    num_data_points = 100
    predict_profile = 'low_precision'
    _, _, losses, predicted_stances, predicted_variances, _, _, _, _ = get_error(user_stance, user_stance_variance, num_data_points, predict_profile=predict_profile, method=method)
    fig, ax = plt.subplots()
    ax.plot(losses, color="red")
    ax1 = ax.twinx()
    if method == "SVI":
        ax1.errorbar(np.arange(len(predicted_stances)), predicted_stances, yerr=predicted_variances, label="Predicted stance")
    elif method == "SVI_categorical":
        ax1.matshow(np.array(predicted_stances).T, extent=[0, len(predicted_stances), 1, -1], aspect='auto', vmin=0, vmax=1, alpha=0.5)
    ax1.errorbar(np.arange(len(predicted_stances)), [user_stance] * len(predicted_stances), yerr=[user_stance_variance] * len(predicted_stances), label="True stance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax1.set_ylabel("Predicted stance")
    ax1.legend()
    fig.savefig(os.path.join(figs_dir_path, f"{method}_estimate_loss.png"))

    if True:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 100
        fig_title = "Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}"
        fig_path = os.path.join(figs_dir_path, f"{method}_estimate_sensitivity_to_user_stance.png")
        plot_experiment(user_stance_variance, num_data_points, 'perfect', method, fig_path, fig_title)

        # test sensitivity to user stance with high variance
        user_stance_variance = 1
        num_data_points = 100
        fig_title = "Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}"
        fig_path = os.path.join(figs_dir_path, f"{method}_estimate_sensitivity_to_user_stance_high_variance.png")
        plot_experiment(user_stance_variance, num_data_points, 'perfect', method, fig_path, fig_title)

    if True:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 2
        fig_title = "Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}"
        fig_path = os.path.join(figs_dir_path, f"{method}_estimate_sensitivity_to_num_data_points.png")
        plot_experiment(user_stance_variance, num_data_points, 'perfect', method, fig_path, fig_title)

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        fig_title = "Precision: {precision}, Recall: {recall}"
        fig_path = os.path.join(figs_dir_path, f"{method}_estimate_sensitivity_to_precision.png")
        plot_experiment(user_stance_variance, num_data_points, 'low_precision', method, fig_path, fig_title)

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        fig_title = "Precision: {precision}, Recall: {recall}"
        fig_path = os.path.join(figs_dir_path, f"{method}_estimate_sensitivity_to_recall.png")
        plot_experiment(user_stance_variance, num_data_points, 'low_recall', method, fig_path, fig_title)


if __name__ == '__main__':
    main()