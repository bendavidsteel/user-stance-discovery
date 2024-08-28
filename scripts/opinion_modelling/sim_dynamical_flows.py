
import gpytorch
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
import tqdm

from mining import estimate
from mining.datasets import GenerativeOpinionTimelineDataset
from mining.generative import SocialGenerativeModel, UpdateDist


# anim = FuncAnimation(fig, ud, frames=num_steps//interval, interval=10, blit=True)
# HTML(anim.to_jshtml())





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
    x_start = torch.min(X) - 0.1 * (torch.max(X) - torch.min(X))
    x_end = torch.max(X) + 0.1 * (torch.max(X) - torch.min(X))
    artists = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        observed_data_artists = ax.plot(X.numpy(), y.numpy(), "kx")
        artists.extend(observed_data_artists)
    if plot_predictions:
        Xtest = torch.linspace(x_start, x_end, n_test)  # test inputs
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
        Xtest = torch.linspace(x_start, x_end, n_test)  # test inputs
        noise = (
            model.noise
            if type(model) not in [gp.models.VariationalSparseGP, gp.models.VariationalGP]
            else model.likelihood.variance
        )
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        samples_artist = ax.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)
        artists.append(samples_artist)

    ax.set_xlim(x_start, x_end)
    return artists

def truncate(times, sequences):
    # hack to get nans
    nan_idxs = torch.where(times != times)[0]
    if len(nan_idxs) == 0:
        return times, sequences
    nan_idx = nan_idxs[0]
    return times[:nan_idx], sequences[:nan_idx]

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():
    num_steps = 500
    num_users = 100
    social_context = SocialGenerativeModel(
        num_users=num_users, 
        sbc_exponent_loc=1.0,
        sbc_exponent_scale=0.1,
        comment_prob=0.2, 
        sus_loc=0.8, 
        sus_scale=0.1,
        seen_att_loc=0.001,
        reply_att_loc=0.1,
        post_att_loc=0.05,
        content_scale=0.1
    )
    interval = 10
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    ud = UpdateDist(social_context, axes[0], axes[1], num_steps=num_steps, interval=interval)
    anim = FuncAnimation(fig, ud, frames=num_steps//interval, interval=10, blit=True)
    anim.save('./figs/flows/gen.mp4')
    
    dataset = GenerativeOpinionTimelineDataset(generative_model=social_context)

    opinion_times, opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=dataset.max_time_step)
    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])
    X = torch.tensor(opinion_times).float()
    max_x = torch.max(torch.nan_to_num(X, 0))
    X_norm = X / max_x
    y = torch.tensor(opinion_sequences).float()

    # limit users for now
    limit_users = 10
    X = X[:limit_users]
    y = y[:limit_users]

    num_users = X.shape[0]
    num_opinions = y.shape[-1]
    num_timesteps = 500
    means = np.full((num_users, num_timesteps, num_opinions), np.nan)

    models = []
    likelihoods = []
    # TODO batch with https://docs.gpytorch.ai/en/latest/examples/07_Pyro_Integration/index.html
    for i in tqdm.tqdm(range(num_users), "Fitting GPs to users"):
        for j in range(num_opinions):
            train_x, train_y = truncate(X_norm[i,:], y[i,:,j])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_y, likelihood)
            models.append(model)
            likelihoods.append(likelihood)

    model_list = gpytorch.models.IndependentModelList(*models)
    likelihood_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)

    model_list.train()
    likelihood_list.train()
    optimizer = torch.optim.Adam(model_list.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood_list, model_list)
    
    losses = []
    training_iter = 50
    for k in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_list(*model_list.train_inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, model_list.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
        losses.append(loss.item())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")  # supress output text
    fig.savefig(f'./figs/flows/losses.png')

    for i in range(num_users):
        for j in range(num_opinions):
            model = model_list.models[i*num_opinions+j]
            likelihood = likelihood_list.likelihoods[i*num_opinions+j]

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            train_x, train_y = truncate(X_norm[i,:], y[i,:,j])
            x_start = max(torch.min(train_x) - 0.1 * (torch.max(train_x) - torch.min(train_x)), 0.)
            x_end = min(torch.max(train_x) + 0.1 * (torch.max(train_x) - torch.min(train_x)), 1.)
            n_test = int((x_end - x_start) * num_timesteps)
            test_x = torch.linspace(x_start, x_end, n_test)  # test inputs

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))

            with torch.no_grad():
                # Initialize plot
                f, ax = plt.subplots(1, 1, figsize=(4, 3))

                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()
                # Plot training data as black stars
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
                # Plot predictive means as blue line
                mean = observed_pred.mean.numpy()
                ax.plot(test_x.numpy(), mean, 'b')
                # Shade between the lower and upper confidence bounds
                ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                ax.set_ylim([-3, 3])
                ax.legend(['Observed Data', 'Mean', 'Confidence'])
                f.savefig(f'./figs/flows/user_{i}_op_{j}_pred.png')


            start_idx = int(x_start * num_timesteps)
            means[i, start_idx:start_idx+mean.shape[0], j] = mean

    fig, ax = plt.subplots()

    for i in range(num_users):
        ax.plot(means[i,:,0], means[i,:,1])

    fig.savefig('./figs/flows/flows.png')
    plt.show()

            
if __name__ == '__main__':
    main()

