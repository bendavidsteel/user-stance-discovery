
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
        sus_loc=0.99, 
        sus_scale=0.1,
        seen_att_loc=0.005,
        reply_att_loc=0.1,
        post_att_loc=0.05,
        content_scale=0.2
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
    # TODO batch with https://docs.gpytorch.ai/en/latest/examples/07_Pyro_Integration/index.html
    for i in tqdm.tqdm(range(num_users), "Fitting GPs to users"):
        for j in range(num_opinions):
            kernel = gp.kernels.RBF(
                input_dim=1, variance=torch.tensor(0.1), lengthscale=torch.tensor(2)
            )
            X_single, y_single = truncate(X_norm[i,:], y[i,:,j])
            gpr = gp.models.GPRegression(X_single, y_single, kernel, noise=torch.tensor(0.5))
            optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
            losses = []
            variances = []
            lengthscales = []
            noises = []
            num_steps = 2000
            for _ in range(num_steps):
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
            fig.savefig(f'./figs/flows/user_{i}_op_{j}_losses.png')

            fig, ax = plt.subplots()
            plot(X_single, y_single, model=gpr, plot_observed_data=True, plot_predictions=True, ax=ax)
            fig.savefig(f'./figs/flows/user_{i}_op_{j}_mean.png')

            x_start = max(torch.min(X_single) - 0.1 * (torch.max(X_single) - torch.min(X_single)), 0.)
            x_end = min(torch.max(X_single) + 0.1 * (torch.max(X_single) - torch.min(X_single)), 1.)
            n_test = int((x_end - x_start) * num_timesteps)
            Xtest = torch.linspace(x_start, x_end, n_test)  # test inputs
            # compute predictive mean and variance
            with torch.no_grad():
                if type(gpr) == gp.models.VariationalSparseGP or type(gpr) == gp.models.VariationalGP:
                    mean, cov = gpr(Xtest, full_cov=True)
                else:
                    mean, cov = gpr(Xtest, full_cov=True, noiseless=False)

            start_idx = int(x_start * num_timesteps)
            means[i, start_idx:start_idx+mean.shape[0], j] = mean

    fig, ax = plt.subplots()

    for i in range(num_users):
        ax.plot(means[i,:,0], means[i,:,1])

    fig.savefig('./figs/flows/flows.png')
    plt.show()

            
if __name__ == '__main__':
    main()

