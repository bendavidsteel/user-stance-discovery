import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch

from mining import estimate, datasets


def plot(
    X,
    y,
    plot_observed_data=False,
    plot_predictions=False,
    n_prior_samples=0,
    model=None,
    kernel=None,
    n_test=500,
    ax=None,
):
    artists = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        observed_data_artists = ax.plot(X.numpy(), y.numpy(), "kx")
        artists.extend(observed_data_artists)
    if plot_predictions:
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP or type(model) == gp.models.VariationalGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        predictions_artists = ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        artists.extend(predictions_artists)
        fill_artist = ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )
        artists.append(fill_artist)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (
            model.noise
            if type(model) != gp.models.VariationalSparseGP
            else model.likelihood.variance
        )
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        samples_artist = ax.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)
        artists.append(samples_artist)

    ax.set_xlim(-0.5, 5.5)
    return artists

def main():
    num_data_points = 10
    user_stance = np.linspace(-1, 1, num_data_points)
    user_stance_variance = 0.1
    predict_profile = 'perfect'
    dataset = datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type=predict_profile, halflife=10.)
    user_times, opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=dataset.max_time_step)
    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])
    X = torch.tensor(user_times).float().view((-1,))
    y = torch.tensor(opinion_sequences[0, :, 0]).float().view((-1,)) + 1

    # def predictor_probs(f):
    #     comment_stance_cats = torch.zeros(len(f), dtype=torch.int)
    #     comment_stance_cats[f > 1/2] = 2
    #     comment_stance_cats[f < -1/2] = 0
    #     comment_stance_cats[(f >= -1/2) & (f <= 1/2)] = 1

    #     # Get prediction probabilities based on the confusion matrix
    #     # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
    #     predict_probs = estimator.predictor_confusion_probs[dataset.stance_columns[0]]['predict_probs'][0, comment_stance_cats, :]
    #     return predict_probs

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    fig_dir_path = os.path.join(this_dir_path, '..', '..', 'figs')

    kernel = gp.kernels.RBF(
        input_dim=1, variance=torch.tensor(6.0), lengthscale=torch.tensor(1)
    )
    likelihood = gp.likelihoods.MultiClass(num_classes=3)#, response_function=predictor_probs)
    gpr = gp.models.VariationalGP(X, y, kernel, likelihood, latent_shape=torch.Size([3]),)

    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    variances = []
    lengthscales = []
    noises = []
    num_steps = 2000
    for i in range(num_steps):
        variances.append(gpr.kernel.variance.item())
        lengthscales.append(gpr.kernel.lengthscale.item())
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")  # supress output text
    fig.savefig(os.path.join(fig_dir_path, 'gp_loss.png'))

    fig, ax = plt.subplots(figsize=(12, 6))
    # kernel_iter = gp.kernels.RBF(
    #     input_dim=1,
    #     variance=torch.tensor(variances[0]),
    #     lengthscale=torch.tensor(lengthscales[0]),
    # )
    # gpr_iter = gp.models.VariationalGP(
    #     X, y, kernel_iter, likelihood
    # )
    # observed_data_artist, predictions_artist = plot(X, y, model=gpr_iter, plot_observed_data=True, plot_predictions=True, ax=ax)

    def update(iteration):
        pyro.clear_param_store()
        ax.cla()
        kernel_iter = gp.kernels.RBF(
            input_dim=1,
            variance=torch.tensor(variances[iteration]),
            lengthscale=torch.tensor(lengthscales[iteration]),
        )
        gpr_iter = gp.models.VariationalGP(
            X, y, kernel_iter, likelihood
        )
        artists = plot(
            X, 
            y, 
            model=gpr_iter, 
            plot_observed_data=True, 
            plot_predictions=True, 
            ax=ax
        )
        ax.set_title(f"Iteration: {iteration}, Loss: {losses[iteration]:0.2f}")
        return artists

    anim = FuncAnimation(fig, update, frames=np.arange(0, num_steps, 30), interval=100, blit=True)
    plt.close()

    anim.save(os.path.join(fig_dir_path, 'gp.gif'), fps=60)

if __name__ == "__main__":
    main()