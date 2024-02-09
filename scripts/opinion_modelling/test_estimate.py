import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
import scipy
import torch
import tqdm

import opinion_datasets, estimate


def get_error(user_stance, user_stance_variance, num_data_points, predict_profile='perfect', model_type="normal", method='SVI', mle=True):
    dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type=predict_profile)
    max_time_step = dataset.max_time_step
    opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=max_time_step)

    mask = torch.tensor(~np.isnan(opinion_sequences[:,:,0])).bool()
    opinion_sequences = torch.tensor(opinion_sequences[:,:,0]).int() + 1
    predictor_ids = torch.tensor(classifier_indices).int()

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])

    if model_type == "normal":
        model = estimator.model
        guide = estimator.guide
        def get_predicted_stats():
            if mle:
                return pyro.param("user_stance").detach().numpy(), pyro.param("user_stance_var").detach().numpy()
            else:
                return pyro.param("user_stance_loc_q").detach().numpy(), pyro.param("user_stance_var_q").detach().numpy()
    elif model_type == "beta":
        model = estimator.beta_model
        guide = estimator.beta_guide
        def get_predicted_stats():
            return pyro.param("alpha").detach().numpy(), pyro.param("beta").detach().numpy()
        
    elif model_type == "categorical":
        model = estimator.categorical_model
        guide = estimator.categorical_guide
        def get_predicted_stats():
            return pyro.param("user_stance").detach().numpy()

    learned_stats = []

    pyro.get_param_store().clear()
    if method == "SVI":
        num_steps = 1000
        gamma = 0.5  # final learning rate will be gamma * initial_lr
        initial_lr = 0.01
        lrd = gamma ** (1 / num_steps)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
        svi = SVI(model, guide, optim, loss=Trace_ELBO())
        losses = []
        for step in tqdm.trange(num_steps):
            loss = svi.step(opinion_sequences, predictor_ids, mask, prior=not mle)
            losses.append(loss)
            if step % 100 == 0:
                learned_stats.append(get_predicted_stats())
        predicted_stats = get_predicted_stats()

    elif method == "MCMC":
        nuts_kernel = pyro.infer.mcmc.NUTS(model)
        mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=100, warmup_steps=50)
        mcmc.run(opinion_sequences, predictor_ids, mask, prior=True)
        losses = []
        for sample in mcmc.get_samples():
            losses.append(0)
        predicted_stance = get_stance()
        predicted_variance = get_variance()
    
    dataset.aggregation = "weighted_mean"
    freq_stats, _ = dataset.get_data(start=0, end=max_time_step)
    freq_mean, freq_variance = freq_stats
    freq_mean, freq_variance = freq_mean[:,0], freq_variance[:,0]
    dataset.aggregation = None
    precision = dataset.all_classifier_profiles['stance_0'][0]['macro']['precision']
    recall = dataset.all_classifier_profiles['stance_0'][0]['macro']['recall']

    return predicted_stats, losses, learned_stats, freq_mean, freq_variance, precision, recall

def plot_experiment(user_stance_variance, num_data_points, predict_profile, model_type, method, mle, fig_path, fig_title):
    # user_stances = []
    # predicted_stances = []
    # predicted_variances = []
    # freq_stances = []
    # freq_variances = []
    user_stances = np.arange(-1, 1, 0.1)
    user_stances = np.tile(np.expand_dims(user_stances, -1), num_data_points)
    # for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
    predicted_stats, _, _, freq_stances, freq_variances, precision, recall = get_error(user_stances, user_stance_variance, num_data_points, predict_profile=predict_profile, model_type=model_type, method=method, mle=mle)
        # user_stances.append(user_stance)
        # predicted_stances.append(predicted_stance)
        # predicted_variances.append(predicted_variance)
        # freq_stances.append(freq_mean)
        # freq_variances.append(freq_var)

    fig, ax = plt.subplots()
    user_stances = user_stances[:,0]
    ax.plot(user_stances, user_stances, label="True stance")
    if model_type == "normal":
        predicted_stances = predicted_stats[0]
        predicted_variances = predicted_stats[1]
        ax.errorbar(user_stances + 0.005, predicted_stances[:,0], yerr=predicted_variances[:,0], label="Predicted stance")
        ax.errorbar(user_stances - 0.005, freq_stances, yerr=freq_variances, label="Frequentist stance")
    elif model_type == "beta":
        predicted_stats = list(zip(*[x[:,0] for x in predicted_stats]))
        use_mean = True
        if use_mean:
            predicted_avg = np.array([scipy.stats.beta.mean(alpha, beta) for alpha, beta in predicted_stats])
            predicted_error = np.array([scipy.stats.beta.std(alpha, beta) for alpha, beta in predicted_stats])
            predicted_avg = predicted_avg * 2 - 1
            predicted_error = predicted_error * 2
        else:
            predicted_avg = np.array([scipy.stats.beta.ppf(0.5, alpha, beta) for alpha, beta in predicted_stats])
            predicted_quartiles = np.array([[scipy.stats.beta.ppf(0.25, alpha, beta), scipy.stats.beta.ppf(0.75, alpha, beta)] for alpha, beta in predicted_stats]).T
            predicted_avg = predicted_avg * 2 - 1

            predicted_quartiles = predicted_quartiles * 2 - 1
            predicted_quartiles[0,:] = predicted_avg - predicted_quartiles[0,:]
            predicted_quartiles[1,:] = predicted_quartiles[1,:] - predicted_avg
            predicted_error = predicted_quartiles
        ax.errorbar(user_stances + 0.005, predicted_avg, yerr=predicted_error, label="Predicted stance")
        ax.errorbar(user_stances - 0.005, freq_stances, yerr=freq_variances, label="Frequentist stance")
    elif model_type == "categorical":
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
    figs_dir_path = os.path.join(root_dir_path, "figs", "estimate")
    
    model_type = "beta"
    mle = True
    method = "SVI"

    # plot loss
    user_stance = -1.0
    user_stance_variance = 0.5
    num_data_points = 100
    predict_profile = 'low_precision'
    _, losses, learned_stats, _, _, _, _ = get_error(user_stance, user_stance_variance, num_data_points, predict_profile=predict_profile, model_type=model_type, method=method, mle=mle)
    fig, ax = plt.subplots()
    ax.plot(losses, color="red")
    ax1 = ax.twinx()
    if model_type == "normal":
        predicted_stances = [x[0][0][0] for x in learned_stats]
        predicted_variances = [x[1][0][0] for x in learned_stats]
        ax1.errorbar(np.linspace(0, 1000, len(predicted_stances)), predicted_stances, yerr=predicted_variances, label="Predicted stance")
    elif model_type == "beta":
        # for i in range(len(learned_stats)):
        #     num_points = 100
        #     x = np.full(num_points, i)
        #     y = np.linspace(-1, 1, num_points)
        #     points = np.array([x, y]).T.reshape(-1, 1, 2)
        #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #     norm = plt.Normalize(0, 1)
        #     lc = LineCollection(segments, cmap='viridis', norm=norm)
        #     # Set the values used for colormapping
        #     alpha, beta = learned_stats[i]
        #     beta_pdf = scipy.stats.beta.pdf(np.linspace(0.001,0.999,num_points-1), alpha, beta)
        #     lc.set_array(beta_pdf)
        #     lc.set_linewidth(2)
        #     line = ax1.add_collection(lc)
        # # fig.colorbar(line, ax=ax1)
        # predicted_avg = np.array([scipy.stats.beta.ppf(0.5, alpha, beta) for alpha, beta in learned_stats]).squeeze(1).squeeze(1)
        use_mean = True
        if use_mean:
            predicted_avg = np.array([scipy.stats.beta.mean(alpha, beta) for alpha, beta in learned_stats]).squeeze(1).squeeze(1)
            predicted_error = np.array([scipy.stats.beta.std(alpha, beta) for alpha, beta in learned_stats]).squeeze(1).squeeze(1)
            predicted_avg = predicted_avg * 2 - 1
            predicted_error = predicted_error * 2
        else:
            predicted_avg = np.array([scipy.stats.beta.ppf(0.5, alpha, beta) for alpha, beta in learned_stats]).squeeze(1).squeeze(1)
            predicted_quartiles = np.array([[scipy.stats.beta.ppf(0.25, alpha, beta), scipy.stats.beta.ppf(0.75, alpha, beta)] for alpha, beta in learned_stats]).squeeze(2).squeeze(2).T
            predicted_avg = predicted_avg * 2 - 1
            predicted_quartiles = predicted_quartiles * 2 - 1
            predicted_quartiles[0,:] = predicted_avg - predicted_quartiles[0,:]
            predicted_quartiles[1,:] = predicted_quartiles[1,:] - predicted_avg
            predicted_error = predicted_quartiles
        ax1.errorbar(np.linspace(0, 1000, len(learned_stats)), predicted_avg, yerr=predicted_error, label="Predicted stance")
    elif model_type == "categorical":
        ax1.matshow(np.array(predicted_stances).squeeze(1).T, extent=[0, len(predicted_stances), 1, -1], aspect='auto', vmin=0, vmax=1, alpha=0.5)
    ax1.errorbar(np.linspace(0, 1000, len(learned_stats)), [user_stance] * len(learned_stats), yerr=[user_stance_variance] * len(learned_stats), label="True stance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax1.set_ylabel("Predicted stance")
    ax1.legend()
    fig.savefig(os.path.join(figs_dir_path, f"{model_type}_estimate_loss.png"))

    if True:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 100
        fig_title = "Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}"
        fig_path = os.path.join(figs_dir_path, f"{model_type}_estimate_sensitivity_to_user_stance.png")
        plot_experiment(user_stance_variance, num_data_points, 'perfect', model_type, method, mle, fig_path, fig_title)

        # test sensitivity to user stance with high variance
        user_stance_variance = 1
        num_data_points = 100
        fig_title = "Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}"
        fig_path = os.path.join(figs_dir_path, f"{model_type}_estimate_sensitivity_to_user_stance_high_variance.png")
        plot_experiment(user_stance_variance, num_data_points, 'perfect', model_type, method, mle, fig_path, fig_title)

    if True:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 2
        fig_title = "Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}"
        fig_path = os.path.join(figs_dir_path, f"{model_type}_estimate_sensitivity_to_num_data_points.png")
        plot_experiment(user_stance_variance, num_data_points, 'perfect', model_type, method, mle, fig_path, fig_title)

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        fig_title = "Precision: {precision}, Recall: {recall}"
        fig_path = os.path.join(figs_dir_path, f"{model_type}_estimate_sensitivity_to_precision.png")
        plot_experiment(user_stance_variance, num_data_points, 'low_precision', model_type, method, mle, fig_path, fig_title)

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        fig_title = "Precision: {precision}, Recall: {recall}"
        fig_path = os.path.join(figs_dir_path, f"{model_type}_estimate_sensitivity_to_recall.png")
        plot_experiment(user_stance_variance, num_data_points, 'low_recall', model_type, method, mle, fig_path, fig_title)


if __name__ == '__main__':
    main()