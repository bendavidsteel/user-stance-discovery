import os

import matplotlib.pyplot as plt
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
import tqdm

import opinion_datasets, estimate


def get_error(user_stance, user_stance_variance, num_data_points, test_points, predict_profile='perfect'):
    dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type=predict_profile, halflife=10.)
    max_time_step = dataset.max_time_step

    predicted_stances = []
    predicted_variances = []
    freq_stances = []
    freq_variances = []
    ends = []

    for end in test_points:
        ends.append(end)
        opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=end)
        data = []
        for i in range(opinion_sequences.shape[1]):
            if not np.isnan(opinion_sequences[0, i, 0]):
                data.append({
                    'predicted_stance': opinion_sequences[0, i, 0],
                    'predictor_id': classifier_indices[0, i]
                })

        estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
        estimator.set_stance(dataset.stance_columns[0])

        method = "SVI"

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
            
            for step in range(num_steps):
                loss = svi.step(data)
                losses.append(loss)

            predicted_stances.append(pyro.param("user_stance").item())
            predicted_variances.append(pyro.param("user_stance_var").item())
        
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


            predicted_stances.append(samples['user_stance_loc'].mean().item())
            predicted_variances.append(samples['user_stance_var'].mean().item())
        
        dataset.aggregation = 'weighted_exponential_smoothing'
        freq_mean, freq_variance, users = dataset.get_data(start=0, end=end)
        dataset.aggregation = None
        freq_stances.append(freq_mean[0,0])
        freq_variances.append(freq_variance[0,0])
    
    precision = dataset.all_classifier_profiles['stance_0'][0]['macro']['precision']
    recall = dataset.all_classifier_profiles['stance_0'][0]['macro']['recall']

    return predicted_stances, predicted_variances, freq_stances, freq_variances, precision, recall

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")
    figs_dir_path = os.path.join(root_dir_path, "figs")

    if True:
        # test constant stance
        user_stance_variance = 0.5
        num_data_points = 100
        fig, axes = plt.subplots(nrows=3)

        for i, user_stance_start in tqdm.tqdm(enumerate([-0.6, 0., 0.6])):
            user_stance = [user_stance_start] * num_data_points
            test_points = list(range(0, num_data_points + 1, num_data_points // 5))
            predicted_stances, predicted_variances, freq_stances, freq_variances, precision, recall = get_error(user_stance, user_stance_variance, num_data_points, test_points)
            axes[i].plot(np.arange(num_data_points), user_stance, label="True stance")
            axes[i].errorbar(np.array(test_points) - num_data_points / 100, predicted_stances, yerr=predicted_variances, label="Predicted stance")
            axes[i].errorbar(np.array(test_points) + num_data_points / 100, freq_stances, yerr=freq_variances, label="Frequentist stance")
            axes[i].set_xlabel("True stance")
            axes[i].set_ylabel("Predicted stance")
            axes[i].legend()

        axes[0].set_title(f"Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}")
        fig.savefig(os.path.join(figs_dir_path, "test_move_estimate_constant.png"))

        # test moving stance
        user_stance_variance = 1
        num_data_points = 100
        fig, axes = plt.subplots(nrows=2)
        for i, (user_stance_start, user_stance_end) in tqdm.tqdm(enumerate([(-0.6, 0.6), (0.6, -0.6)])):
            user_stance = np.linspace(user_stance_start, user_stance_end, num_data_points)
            test_points = list(range(0, num_data_points + 1, num_data_points // 5))
            predicted_stances, predicted_variances, freq_stances, freq_variances, precision, recall = get_error(user_stance, user_stance_variance, num_data_points, test_points)
            axes[i].plot(np.arange(num_data_points), user_stance, label="True stance")
            axes[i].errorbar(np.array(test_points) - num_data_points / 100, predicted_stances, yerr=predicted_variances, label="Predicted stance")
            axes[i].errorbar(np.array(test_points) + num_data_points / 100, freq_stances, yerr=freq_variances, label="Frequentist stance")
            axes[i].set_xlabel("True stance")
            axes[i].set_ylabel("Predicted stance")
            axes[i].legend()

        axes[0].set_title(f"Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}")
        fig.savefig(os.path.join(figs_dir_path, "test_move_estimate_moving.png"))

    if False:
        # test sensitivity to user stance
        user_stance_variance = 0.5
        num_data_points = 2
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        freq_stances = []
        freq_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _, freq_mean, freq_var, _, _ = get_error(user_stance, user_stance_variance, num_data_points)
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)
            freq_stances.append(freq_mean)
            freq_variances.append(freq_var)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")
        ax.errorbar(user_stances, freq_stances, yerr=freq_variances, label="Frequentist stance")
        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.set_title(f"Num Data Points: {num_data_points}, User Stance Variance: {user_stance_variance}")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_num_data_points.png"))

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        freq_stances = []
        freq_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _, freq_mean, freq_var, precision, recall = get_error(user_stance, user_stance_variance, num_data_points, predict_profile='low_precision')
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)
            freq_stances.append(freq_mean)
            freq_variances.append(freq_var)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")
        ax.errorbar(user_stances, freq_stances, yerr=freq_variances, label="Frequentist stance")
        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.set_title(f"Precision: {precision}, Recall: {recall}")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_precision.png"))

        # test sensitivity to user stance with high variance
        user_stance_variance = 0.5
        num_data_points = 100
        user_stances = []
        predicted_stances = []
        predicted_variances = []
        freq_stances = []
        freq_variances = []
        for user_stance in tqdm.tqdm(np.arange(-1, 1, 0.1)):
            predicted_stance, predicted_variance, _, _, _, freq_mean, freq_var, precision, recall = get_error(user_stance, user_stance_variance, num_data_points, predict_profile='low_recall')
            user_stances.append(user_stance)
            predicted_stances.append(predicted_stance)
            predicted_variances.append(predicted_variance)
            freq_stances.append(freq_mean)
            freq_variances.append(freq_var)

        fig, ax = plt.subplots()
        ax.plot(user_stances, user_stances, label="True stance")
        ax.errorbar(user_stances, predicted_stances, yerr=predicted_variances, label="Predicted stance")
        ax.errorbar(user_stances, freq_stances, yerr=freq_variances, label="Frequentist stance")
        ax.set_xlabel("True stance")
        ax.set_ylabel("Predicted stance")
        ax.set_title(f"Precision: {precision}, Recall: {recall}")
        ax.legend()
        fig.savefig(os.path.join(figs_dir_path, "test_estimate_sensitivity_to_recall.png"))



if __name__ == '__main__':
    main()