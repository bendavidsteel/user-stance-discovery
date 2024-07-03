import collections
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import estimate, opinion_datasets

def plot_movement(dataset, root_dir_path):
    num_people = dataset.num_people

    # opinion_timelines = np.full((dataset.max_time_step, num_people, dataset.num_opinions), np.nan)
    # for i in tqdm.tqdm(range(dataset.max_time_step)):
    #     opinion_mean, opinion_variance, opinion_timeline = dataset[i]
    #     opinion_timelines[i] = opinion_mean
    min_time_step = dataset.min_time_step.to_pydatetime()
    max_time_step = dataset.max_time_step.to_pydatetime()
    # get all months between min and max time step
    starts = []
    ends = []
    start_year = min_time_step.year
    start_month = min_time_step.month
    start = datetime.datetime(start_year, start_month, 1)
    while start < max_time_step:
        if start.month == 12:
            end = datetime.datetime(start.year + 1, 1, 1)
        else:
            end = datetime.datetime(start.year, start.month + 1, 1)
        starts.append(start)
        ends.append(end)
        start = end

    opinion_timelines, opinion_timelines_var, users = dataset.get_data(starts, ends)

    top_num = 3

    # get max movement for each stance
    max_opinion = np.nanmax(opinion_timelines, axis=0)
    min_opinion = np.nanmin(opinion_timelines, axis=0)
    max_opinion_movement = max_opinion - min_opinion

    # plot top 5 movers for each stance
    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), figsize=(10, 2.5*len(dataset.stance_columns)))
    for i in range(len(dataset.stance_columns)):
        max_stance_movement = max_opinion_movement[:, i]
        non_nans = ~np.isnan(max_stance_movement)
        max_stance_movement = max_stance_movement[non_nans]
        stance_timelines = opinion_timelines[:, non_nans, i]
        stance_timelines_var = opinion_timelines_var[:, non_nans, i]
        sorted_stance_movement = np.sort(max_stance_movement)
        sorted_stance_movers = np.argsort(max_stance_movement)
        top_stance_movement = sorted_stance_movement[-top_num:]
        top_stance_movers = sorted_stance_movers[-top_num:]
        for j in range(top_num):
            axes[i].errorbar(ends, stance_timelines[:, top_stance_movers[-j-1]], yerr=stance_timelines_var[:, top_stance_movers[-j-1]])
        axes[i].set_title(dataset.stance_columns[i].replace('stance_', '').replace('_', ' ').title())
    fig.savefig(os.path.join(root_dir_path, "figs", "top_stance_movers.png"))
    

    # get max movement for all stances
    max_movement = np.zeros((opinion_timelines.shape[1],))
    max_movement_indices = np.zeros((opinion_timelines.shape[1], 2))
    for i in range(opinion_timelines.shape[0]):
        for j in range(i+1, opinion_timelines.shape[0]):
            movement = np.sqrt(np.sum((opinion_timelines[i] - opinion_timelines[j])**2, axis=1))
            max_movement = np.select([movement > max_movement], [movement], default=max_movement)
            max_movement_indices = np.select(
                [np.broadcast_to((movement > max_movement), (2, max_movement.shape[0])).T], 
                [np.broadcast_to(np.array([i, j]), max_movement_indices.shape)], 
                default=max_movement_indices
            )

    sorted_max_movement = np.sort(max_movement)
    sorted_max_movers = np.argsort(max_movement)
    top_movement = sorted_max_movement[-top_num:]
    top_movers = sorted_max_movers[-top_num:]

    # plot top 5 movers
    # map users to colors
    user_to_color = {i: plt.cm.tab10(i) for i in range(5)}
    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), figsize=(10, 2.5*len(dataset.stance_columns)))
    for i in range(len(dataset.stance_columns)):
        for j in range(top_num):
            axes[i].errorbar(ends, opinion_timelines[:, top_movers[-j-1], i], yerr=opinion_timelines_var[:, top_movers[-j-1], i], color=user_to_color[j])
        axes[i].set_title(dataset.stance_columns[i].replace('stance_', '').replace('_', ' ').title())
    fig.savefig(os.path.join(root_dir_path, "figs", "top_movers.png"))


    fig, axes = plt.subplots(nrows=len(dataset.stance_columns), figsize=(10, 2.5*len(dataset.stance_columns)))
    for i in range(len(dataset.stance_columns)):
        axes[i].plot(ends, opinion_timelines[:, :, i])
        axes[i].set_title(dataset.stance_columns[i].replace('stance_', '').replace('_', ' ').title())

    fig.savefig(os.path.join(root_dir_path, "figs", "opinion_timelines.png"))

def fit_hmm(dataset):
    from pomegranate.distributions import Categorical
    from pomegranate.hmm import DenseHMM

    dataset.aggregation = None
    opinion_sequences, users, classifier_indices = dataset.get_data(start=dataset.min_time_step, end=dataset.max_time_step)

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)

    for i, stance in enumerate(dataset.stance_columns):
        estimator.set_stance(stance)

        X = []
        classifier_counts = {i: {j: 0 for j in range(len(dataset.all_classifier_profiles[stance]))} for i in range(3)}
        for j in range(opinion_sequences.shape[0]):
            seq = opinion_sequences[j, np.where(~np.isnan(opinion_sequences[j, :, i]))[0], i]
            seq = torch.tensor(seq.astype(int) + 1).reshape((-1, 1))
            if seq.shape[0] > 0:
                X.append(seq)

            for k in range(opinion_sequences.shape[1]):
                if not np.isnan(opinion_sequences[j, k, i]):
                    if opinion_sequences[j, k, i] == -1:
                        classifier_counts[0][classifier_indices[j, k]] += 1
                    elif opinion_sequences[j, k, i] == 0:
                        classifier_counts[1][classifier_indices[j, k]] += 1
                    elif opinion_sequences[j, k, i] == 1:
                        classifier_counts[2][classifier_indices[j, k]] += 1

        # normalize classifier_counts
        for j in range(3):
            j_sum = np.sum(list(classifier_counts[j].values()))
            for k in range(len(dataset.all_classifier_profiles[stance])):
                classifier_counts[j][k] /= j_sum

        # batch sequences by length
        X_dict = collections.defaultdict(list)
        for i, x in enumerate(X):
            n = len(x)
            X_dict[n].append(x)

        keys = sorted(X_dict.keys())
        X = [torch.stack(X_dict[key]) for key in keys]

        # Get prediction probabilities based on the confusion matrix
        # Assume self.predictor_confusion_probs is properly vectorized to handle batched inputs
        confusion_probs = estimator.predictor_confusion_probs[stance]['predict_probs']
        against_predict_probs = torch.sum(torch.tensor([(confusion_probs[c_idx, 0, :] * classifier_counts[0][c_idx]).tolist() for c_idx in classifier_counts[0]]), axis=0)
        neutral_predict_probs = torch.sum(torch.tensor([(confusion_probs[c_idx, 1, :] * classifier_counts[1][c_idx]).tolist() for c_idx in classifier_counts[1]]), axis=0)
        favor_predict_probs = torch.sum(torch.tensor([(confusion_probs[c_idx, 2, :] * classifier_counts[2][c_idx]).tolist() for c_idx in classifier_counts[2]]), axis=0)

        against = Categorical([against_predict_probs.tolist()])
        neutral = Categorical([neutral_predict_probs.tolist()])
        favor = Categorical([favor_predict_probs.tolist()])

        edges = [[0.9, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]
        starts = [0.8, 0.1, 0.1]

        model = DenseHMM([against, neutral, favor], edges=edges, starts=starts, verbose=False)
        X_ = torch.cat([x.reshape(-1, 1) for x in X], dim=0).unsqueeze(0) # Add this
        model._initialize(X_) # Add this too
        model.fit(X)

        print(f"Stance: {stance}")
        for i, stance_i in enumerate(["against", "neutral", "favor"]):
            for j, stance_j in enumerate(["against", "neutral", "favor"]):
                print(f"{stance_i} -> {stance_j}: {torch.exp(model.edges[i, j]):.3f}")

def pyro_hmm(dataset, root_dir_path):
    import pyro
    import pyro.distributions as dist
    from pyro import poutine
    from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
    from pyro.infer.autoguide import AutoDelta
    from pyro.util import ignore_jit_warnings

    def model(sequences, classifier_indices, lengths, predictor_probs, emission_probs, num_states=3, jit=True, batch_size=None, include_prior=True):
        with ignore_jit_warnings():
            num_sequences, max_length = map(int, sequences.shape)
            assert lengths.shape == (num_sequences,)
            assert lengths.max() <= max_length
        with poutine.mask(mask=include_prior):
            # start_probs = pyro.sample(
            #     "probs_start",
            #     dist.Dirichlet(torch.tensor([0.5, 0.5])),
            # )
            start_probs = pyro.sample(
                "probs_start",
                dist.Dirichlet(torch.tensor([1.0, 10.0, 1.0]))
            )
            probs_x = pyro.sample(
                "probs_x",
                dist.Dirichlet(0.9 * torch.eye(num_states) + 0.1).to_event(1),
            )
            emission_probs = pyro.sample(
                "probs_emission",
                dist.Dirichlet(torch.tensor([[4.0, 3.0, 2.0], [2.5, 4.0, 2.5], [2.0, 3.0, 4.0]])).to_event(1),
            )

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
                    predict_probs = predictor_probs[classifier_indices[batch, t], x.squeeze(-1), :]
                    # with outputs_plate:
                    pyro.sample(
                        "y_predicted_{}".format(t),
                        dist.Categorical(predict_probs),
                        obs=sequences[batch, t],
                    )

    dataset.aggregation = None
    opinion_sequences, users, classifier_indices = dataset.get_data(start=dataset.min_time_step, end=dataset.max_time_step)

    if 'subreddit' in users.columns:
        subreddits = users['subreddit'].unique()
    else:
        subreddits = [None]

    # dataset.aggregation = "inferred_categorical"
    # opinion_stats, users = dataset.get_data(start=dataset.min_time_step, end=dataset.max_time_step)
    # opinion_categoricals = opinion_stats[0]

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    pyro.set_rng_seed(42)

    num_steps = 1000
    initial_lr = 0.1
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / num_steps)
    jit = False
    batch_size = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_indices = torch.tensor(classifier_indices).to(device)

    for stance_idx, stance in enumerate(dataset.stance_columns):
        for subreddit in subreddits:

            if subreddit:
                print(f"Subreddit: {subreddit}")
                fig_dir_path = os.path.join(root_dir_path, "figs", "hmm", subreddit)
                if not os.path.exists(fig_dir_path):
                    os.makedirs(fig_dir_path)
            else:
                fig_dir_path = os.path.join(root_dir_path, "figs", "hmm")

            probs_path = os.path.join(fig_dir_path, f"{stance}_probs.json")
            if os.path.exists(probs_path):
                continue

            seqs = []
            c_indices = []
            for j in range(opinion_sequences.shape[0]):
                if subreddit:
                    if users['subreddit'][j] != subreddit:
                        continue
                seq = opinion_sequences[j, np.where(~np.isnan(opinion_sequences[j, :, stance_idx]))[0], stance_idx]
                seq = torch.tensor(seq.astype(int) + 1)
                if seq.shape[0] > 0:
                    seqs.append(seq)
                    c_indices.append(classifier_indices[j, np.where(~np.isnan(opinion_sequences[j, :, stance_idx]))[0]])

            if len(seqs) == 0:
                continue

            max_length = max([len(seq) for seq in seqs])
            sequences = torch.zeros((len(seqs), max_length), dtype=torch.int)
            classifier_index = torch.zeros((len(seqs), max_length), dtype=torch.int)
            for i, seq in enumerate(seqs):
                sequences[i, :len(seq)] = seq
                classifier_index[i, :len(seq)] = c_indices[i]

            sequences = sequences.to(device)
            lengths = torch.tensor([len(seq) for seq in seqs]).to(device)
            classifier_index = classifier_index.to(device)
            predictor_probs = estimator.predictor_confusion_probs[stance]['predict_probs'].to(device)
            num_observations = float(lengths.sum())

            num_states = 3

            if num_states == 3:
                emission_probs = torch.tensor([[0.4, 0.4, 0.2], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]]).to(device)
                # how many categoricals fall into these categories?
                # stance_categoricals = opinion_categoricals[:, stance_idx, :]
                # num_extreme_against = len(np.where(stance_categoricals[:, 0] >= emission_probs[0][0].item())[0])
                # num_extreme_neutral = len(np.where(stance_categoricals[:, 1] >= emission_probs[1][1].item())[0])
                # num_extreme_favor = len(np.where(stance_categoricals[:, 2] >= emission_probs[2][2].item())[0])
                # print(f"Stance: {stance}")
                # print(f"Num extreme against: {num_extreme_against}")
                # print(f"Num extreme neutral: {num_extreme_neutral}")
                # print(f"Num extreme favor: {num_extreme_favor}")
                # print(f"Num left over: {len(stance_categoricals) - num_extreme_against - num_extreme_neutral - num_extreme_favor}")
            elif num_states == 2:
                emission_probs = torch.tensor([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4]]).to(device)
            
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
            optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
            elbo = JitTraceEnum_ELBO if jit else TraceEnum_ELBO
            elbo = elbo(
                max_plate_nesting=1 if model.__name__ == 'model_0' else 1,
                strict_enumeration_warning=(True),
            )
            svi = SVI(model, guide, optim, elbo)

            # We'll train on small minibatches.
            pbar = tqdm.tqdm(range(num_steps))
            losses = []
            for step in pbar:
                batch_size = min(batch_size, len(sequences)) if batch_size is not None else None
                loss = svi.step(sequences, classifier_index, lengths, predictor_probs, emission_probs, num_states=num_states, jit=jit, batch_size=batch_size)
                losses.append(loss / num_observations)
                pbar.set_description(f"loss: {loss / num_observations:.4f}")

            # Plot the loss curve.
            fig, ax = plt.subplots()
            ax.plot(losses)
            ax.set(xlabel="Step", ylabel="Loss")
            fig.savefig(os.path.join(fig_dir_path, f"{stance}_loss.png"))

            print(f"Stance: {stance}")
            if num_states == 3:
                for i, stance_i in enumerate(["against", "neutral", "favor"]):
                    for j, stance_j in enumerate(["against", "neutral", "favor"]):
                        prob_i_to_j = pyro.param('AutoDelta.probs_x')[i,j]
                        print(f"{stance_i} -> {stance_j}: {prob_i_to_j:.3f}")
            elif num_states == 2:
                for i, stance_i in enumerate(["against", "favor"]):
                    for j, stance_j in enumerate(["against", "favor"]):
                        prob_i_to_j = pyro.param('AutoDelta.probs_x')[i,j]
                        print(f"{stance_i} -> {stance_j}: {prob_i_to_j:.3f}")

            with open(probs_path, 'w') as f:
                json.dump({k: v.cpu().tolist() for k, v in pyro.get_param_store().items()}, f)

            # look at learned markov process sequences
            # use viterbi to get most likely sequence
            # guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

            # learned_sequence = np.zeros_like(sequences.cpu().numpy())
            # for i in range(sequences.shape[0]):
            #     for t in range(lengths[i]):
            #         learned_sequence[i, t] = pyro.param(f'AutoDelta.x_{t}')[i].argmax().item()

            # # plot learned sequence
            # fig, ax = plt.subplots()
            # for i in range(learned_sequence.shape[0]):
            #     ax.plot(learned_sequence[i, :lengths[i]])
            # fig.savefig(os.path.join(root_dir_path, "figs", "hmm", f"{stance}_learned_sequences.png"))

def compare_hmm(dataset, root_dir_path):
    dataset.aggregation = None
    opinion_sequences, users, classifier_indices = dataset.get_data(start=dataset.min_time_step, end=dataset.max_time_step)

    subreddits = users['subreddit'].unique()

    # dataset.aggregation = "inferred_categorical"
    # opinion_stats, users = dataset.get_data(start=dataset.min_time_step, end=dataset.max_time_step)
    # opinion_categoricals = opinion_stats[0]

    estimator = estimate.StanceEstimation(dataset.all_classifier_profiles)

    def get_f_norm(fig_dir_path, stance, subreddit):
        if not os.path.exists(os.path.join(fig_dir_path, f"{stance}_probs.json")):
            raise FileNotFoundError(f"{stance}_probs.json not found")
        with open(os.path.join(fig_dir_path, f"{stance}_probs.json"), 'r') as f:
            params = json.load(f)

        transition_probs = np.array(params['AutoDelta.probs_x'])
        identity_probs = np.eye(transition_probs.shape[0])
        f_norm = np.linalg.norm(np.abs(transition_probs - identity_probs), ord='fro')

        return f_norm

    f_norms = {}
    for stance_idx, stance in enumerate(dataset.stance_columns):
        fig_dir_path = os.path.join(root_dir_path, "figs", "hmm")
        f_norm = get_f_norm(fig_dir_path, stance, None)
        f_norms[stance] = {}
        f_norms[stance]['all'] = f_norm
        for subreddit in subreddits:
            try:
                fig_dir_path = os.path.join(root_dir_path, "figs", "hmm", subreddit)
                f_norm = get_f_norm(fig_dir_path, stance, subreddit)
                f_norms[stance][subreddit] = f_norm
            except:
                pass

    pretty_stances = {s: s.replace('stance_', '').replace('_', ' ').title() for s in dataset.stance_columns}
    mapped = {
        'Ndp': 'NDP',
    }
    pretty_stances = {s: mapped[p] if p in mapped else p for s, p in pretty_stances.items()}

    if all([len(f_norms[stance]) > 1 for stance in f_norms]):
        for stance in f_norms:
            fig, ax = plt.subplots()
            for subreddit in subreddits:
                ax.bar(subreddit, f_norms[stance][subreddit])
            ax.set_title(pretty_stances[stance])
            ax.set_ylabel("Opinion Movement (Frobenius Norm)")
            ax.set_xlabel("Subreddit")
            ax.tick_params(axis='x', labelrotation=45)
            fig.savefig(os.path.join(fig_dir_path, f"{stance}_f_norms.png"))

    fig_dir_path = os.path.join(root_dir_path, "figs", "hmm")
    fig, ax = plt.subplots(figsize=(4, 2))
    for stance in f_norms:
        ax.bar(pretty_stances[stance], f_norms[stance]['all'])
    # ax.set_title("Frobenius Norms")
    ax.set_ylabel("Opinion Movement (Frobenius Norm)")
    ax.set_xlabel("Stance")
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir_path, "f_norms.png"))

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    dataset_name = "reddit"
    aggregation = "weighted_exponential_smoothing"

    if dataset_name == "reddit":
        experiment_name = "1sub_1year"
        if experiment_name == "4sub_1year":
            topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "4sub_1year")
        else:
            topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "1sub_1year", "topics_minilm_0_2")
        dataset = opinion_datasets.RedditOpinionTimelineDataset(topics_dir_path, aggregation=aggregation, halflife=100., min_num_per_stance=50)
    elif dataset_name == "generative":
        num_people = 100
        num_opinions = 3
        num_data_points = 100
        user_stance = np.random.uniform(-1, 1, (num_people, num_opinions))
        user_stances = np.zeros((num_people, num_opinions, num_data_points))
        user_stances[:, :, 0] = user_stance
        for i in range(1, num_data_points):
            user_stances[:, :, i] = user_stances[:, :, i-1] + np.random.normal(0, 0.1, (num_people, num_opinions))
        user_stance_variance = np.tile(np.random.uniform(0, 0.1, (num_people, num_opinions, 1)), num_data_points)
        fig_path = os.path.join(root_dir_path, "figs", "generative")
        dataset = opinion_datasets.SimpleGenerativeOpinionTimelineDataset(user_stances, user_stance_variance, num_data_points, pred_profile_type='low_recall', aggregation=aggregation)

        starts = [0]
        ends = [num_data_points]
    
    do_plot_movement = False
    if do_plot_movement:
        plot_movement(dataset, root_dir_path)

    do_hmm = False
    if do_hmm:
        pyro_hmm(dataset, root_dir_path)

    do_compare_hmm = True
    if do_compare_hmm:
        compare_hmm(dataset, root_dir_path)

if __name__ == "__main__":
    main()