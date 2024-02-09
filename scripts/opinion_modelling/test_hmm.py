import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import estimate, opinion_datasets

def pomegranate_hmm(opinion_sequences, estimator, dataset):
    from pomegranate.distributions import Categorical
    from pomegranate.hmm import DenseHMM

    X = (opinion_sequences + 1).astype(int)

    # Get prediction probabilities based on the confusion matrix
    # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
    against_predict_probs = estimator.predictor_confusion_probs[dataset.stance_columns[0]]['predict_probs'][0, 0, :]
    neutral_predict_probs = estimator.predictor_confusion_probs[dataset.stance_columns[0]]['predict_probs'][0, 1, :]
    favor_predict_probs = estimator.predictor_confusion_probs[dataset.stance_columns[0]]['predict_probs'][0, 2, :]

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    fig_dir_path = os.path.join(this_dir_path, '..', '..', 'figs')

    against = Categorical([against_predict_probs.tolist()])
    neutral = Categorical([neutral_predict_probs.tolist()])
    favor = Categorical([favor_predict_probs.tolist()])

    edges = [[0.9, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]
    starts = [0.8, 0.1, 0.1]

    model = DenseHMM([against, neutral, favor], edges=edges, starts=starts, verbose=True)
    model.fit(X)

    print(model.edges)

def pyro_hmm(opinion_sequences, classifier_indices, estimator, dataset):
    import pyro
    import pyro.distributions as dist
    from pyro import poutine
    from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
    from pyro.infer.autoguide import AutoDelta
    from pyro.optim import Adam
    from pyro.util import ignore_jit_warnings

    def model_0(sequences, classifier_indices, lengths, predictor_probs, num_states=3, jit=True, batch_size=None, include_prior=True):
        with ignore_jit_warnings():
            num_sequences, max_length, data_dim = map(int, sequences.shape)
            assert lengths.shape == (num_sequences,)
            assert lengths.max() <= max_length
        with poutine.mask(mask=include_prior):
            probs_x = pyro.sample(
                "probs_x",
                dist.Dirichlet(0.9 * torch.eye(num_states) + 0.1).to_event(1),
            )

        output_plate = pyro.plate("outputs", data_dim, dim=-1)
        # We subsample batch_size items out of num_sequences items.
        for i in pyro.plate("sequences", num_sequences):
            length = lengths[i]
            sequence = sequences[i, :length, :]
            seq_classifier_indices = classifier_indices[i, :length]
            x = 0
            # If we are not using the jit, then we can vary the program structure
            # each call by running for a dynamically determined number of time
            # steps, lengths.max(). However if we are using the jit, then we try to
            # keep a single program structure for all minibatches; the fixed
            # structure ends up being faster since each program structure would
            # need to trigger a new jit compile stage.
            for t in pyro.markov(range(length)):
                x = pyro.sample(
                    f"x_{i}_{t}",
                    dist.Categorical(probs_x[x]),
                    infer={"enumerate": "parallel"},
                )
                predict_probs = predictor_probs[seq_classifier_indices[t], x.squeeze(-1), :]
                with output_plate:
                    pyro.sample(
                        f"y_{i}_{t}",
                        dist.Categorical(predict_probs.unsqueeze(0)),
                        obs=sequence[t],
                    )

    def model_1(sequences, classifier_indices, lengths, predictor_probs, num_states=3, jit=True, batch_size=None, include_prior=True):
        with ignore_jit_warnings():
            num_sequences, max_length = map(int, sequences.shape)
            assert lengths.shape == (num_sequences,)
            assert lengths.max() <= max_length
        with poutine.mask(mask=include_prior):
            start_probs = pyro.sample(
                "probs_start",
                dist.Dirichlet(torch.tensor([0.1, 0.8, 0.1])),
            )
            probs_x = pyro.sample(
                "probs_x",
                dist.Dirichlet(0.9 * torch.eye(num_states) + 0.1).to_event(1),
            )
        emission_probs = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]).to(device)

        # We subsample batch_size items out of num_sequences items.
        # outputs_plate = pyro.plate("outputs", data_dim, dim=-1)
        with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
            lengths = lengths[batch]
            x = pyro.sample("x_start", dist.Categorical(start_probs), infer={"enumerate": "parallel"})
            # If we are not using the jit, then we can vary the program structure
            # each call by running for a dynamically determined number of time
            # steps, lengths.max(). However if we are using the jit, then we try to
            # keep a single program structure for all minibatches; the fixed
            # structure ends up being faster since each program structure would
            # need to trigger a new jit compile stage.
            for t in pyro.markov(range(max_length if jit else lengths.max())):
                with poutine.mask(mask=(t < lengths)):
                    x = pyro.sample(
                        f"x_{t}",
                        dist.Categorical(probs_x[x]),
                        infer={"enumerate": "parallel"},
                    )
                    y = pyro.sample(
                        f"y_{t}",
                        dist.Categorical(emission_probs[x]),
                        infer={"enumerate": "parallel"},
                    )
                    predict_probs = predictor_probs[classifier_indices[batch, t], y.squeeze(-1), :]
                    # with outputs_plate:
                    pyro.sample(
                        f"obs_y_{t}",
                        dist.Categorical(predict_probs),
                        obs=sequences[batch, t],
                    )

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    pyro.set_rng_seed(42)

    model = model_1
    learning_rate = 0.001
    jit = False
    num_steps = 1000
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_indices = torch.tensor(classifier_indices).to(device)

    for stance_idx, stance in enumerate(dataset.stance_columns):
        sequences = torch.tensor(opinion_sequences[:,:,stance_idx] + 1).to(device)
        lengths = torch.tensor(sequences.shape[1] * np.ones(sequences.shape[0], dtype=int)).to(device)
        predictor_probs = estimator.predictor_confusion_probs[stance]['predict_probs'].to(device)
        num_observations = float(lengths.sum())
        
        pyro.clear_param_store()

        # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
        # out the hidden state x. This is accomplished via an automatic guide that
        # learns point estimates of all of our conditional probability tables,
        # named probs_*.
        guide = AutoDelta(
            poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_"))
        )

        # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
        # All of our models have one plate: "data".
        optim = Adam({"lr": learning_rate})
        elbo = JitTraceEnum_ELBO if jit else TraceEnum_ELBO
        elbo = elbo(
            max_plate_nesting=1 if model.__name__ == 'model_0' else 1,
            strict_enumeration_warning=(True),
        )
        svi = SVI(model, guide, optim, elbo)

        # We'll train on small minibatches.
        pbar = tqdm.tqdm(range(num_steps))
        for step in pbar:
            loss = svi.step(sequences, classifier_indices, lengths, predictor_probs, jit=jit, batch_size=min(batch_size, len(sequences)))
            pbar.set_description(f"loss: {loss / num_observations:.4f}")

        print(f"Stance: {stance}")
        for i, stance_i in enumerate(["against", "neutral", "favor"]):
            for j, stance_j in enumerate(["against", "neutral", "favor"]):
                prob_i_to_j = pyro.param('AutoDelta.probs_x')[i,j]
                print(f"{stance_i} -> {stance_j}: {prob_i_to_j:.3f}")


def main():
    num_data_points = 10
    num_users = 16
    user_stance = np.broadcast_to(np.linspace(-1, 1, num_data_points), (num_users, num_data_points))
    user_stance_variance = 0.1
    predict_profile = 'perfect'
    dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stance, user_stance_variance, num_data_points, pred_profile_type=predict_profile, halflife=10.)
    opinion_sequences, users, classifier_indices = dataset.get_data(start=0, end=dataset.max_time_step)
    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)
    estimator.set_stance(dataset.stance_columns[0])

    method = 'pyro'
    if method == 'pomegranate':
        pomegranate_hmm(opinion_sequences, estimator, dataset)
    elif method == 'pyro':
        pyro_hmm(opinion_sequences, classifier_indices, estimator, dataset)

if __name__ == "__main__":
    main()