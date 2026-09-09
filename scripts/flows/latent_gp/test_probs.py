"""Checks on the soft-evidence observation models.

Two things are pinned here: that the mixture form reduces exactly to the count
form when the classifier is certain, so 'hard' still reproduces every earlier
result, and that it recovers a known latent better than labels do when the
classifier is not. Run the second with `pytest -m slow`.
"""

import sys
import os

import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_gp import ordinal, probs


def corner_index(resolution, ordinal_class):
    """Lattice index of the corner asserting `ordinal_class` (0=neg, 1=neu, 2=pos)."""
    grid = probs.simplex_grid(resolution)[:, probs.TO_ORDINAL]
    return int(np.flatnonzero(grid[:, ordinal_class] == 1.0)[0])


def test_grid_is_a_simplex():
    for r in (2, 3, 4, 6):
        g = probs.simplex_grid(r)
        assert g.shape == (probs.n_archetypes(r), 3)
        assert np.allclose(g.sum(1), 1.0)
        assert len(np.unique(g, axis=0)) == len(g)


def test_resolution_divisible_by_three_holds_the_centroid():
    for r in (3, 6):
        assert (np.abs(probs.simplex_grid(r) - 1 / 3).sum(1) < 1e-12).any()
    for r in (2, 4):
        assert not (np.abs(probs.simplex_grid(r) - 1 / 3).sum(1) < 1e-12).any()


def test_assign_snaps_confident_posts_to_corners():
    q = np.array([[0.99, 0.005, 0.005], [0.01, 0.98, 0.01], [0.02, 0.02, 0.96]])
    idx = probs.assign(q, 6)
    got = probs.simplex_grid(6)[idx]
    assert np.allclose(got, np.eye(3)[[0, 1, 2]])


def test_temper_matches_logit_scaling():
    """Powering probabilities is temperature scaling of the logits."""
    logit = np.array([[2.0, -0.5, 0.3]])
    q = np.exp(logit) / np.exp(logit).sum()
    for T in (0.5, 1.0, 2.0, 4.0):
        direct = np.exp(logit / T) / np.exp(logit / T).sum()
        assert np.allclose(probs.temper(q, T), direct, atol=1e-12)


def test_archetypes_are_distributions_with_no_hard_zeros():
    a = probs.archetypes(6, temperature=1.0, floor=0.01)
    assert np.allclose(a.sum(1), 1.0)
    assert a.min() > 0.0


@pytest.mark.parametrize('resolution', [3, 6])
def test_mixture_reduces_to_counts_on_hard_labels(resolution):
    """One-hot lattice mass gives the count likelihood, up to the pi offset."""
    rng = np.random.default_rng(0)
    n_cells = 64
    c = 0.7
    m = jnp.asarray(rng.normal(0, 1.2, n_cells))
    v = jnp.asarray(rng.uniform(0.05, 0.9, n_cells))

    counts = rng.integers(0, 30, (n_cells, 3)).astype(float)     # neg, neu, pos
    L = probs.n_archetypes(resolution)
    n_arch = np.zeros((n_cells, L))
    for k in range(3):
        n_arch[:, corner_index(resolution, k)] = counts[:, k]

    pi = counts.sum(0) / counts.sum()
    onehot = np.eye(3)[np.array([probs.simplex_grid(resolution)[:, probs.TO_ORDINAL]
                                 .argmax(1)]).ravel()]
    # only the corners carry mass, so the other rows' values never enter
    logL = probs.log_likelihood_ratio(np.maximum(onehot, 1e-300), pi)

    mix = ordinal.mixture_expected_loglik(m, v, c, jnp.asarray(n_arch), jnp.asarray(logL))
    exact = ordinal.expected_loglik(m, v, c, *(jnp.asarray(counts[:, k]) for k in range(3)))
    offset = -(counts * np.log(pi)).sum(1)

    assert np.allclose(np.asarray(mix), np.asarray(exact) + offset, atol=1e-8)


def test_mixture_sites_match_count_sites_on_hard_labels():
    """The site the smoother consumes must be identical, not merely close."""
    rng = np.random.default_rng(1)
    n_cells, r, c = 48, 6, 0.55
    m = jnp.asarray(rng.normal(0, 1.0, n_cells))
    v = jnp.asarray(rng.uniform(0.1, 0.8, n_cells))
    counts = rng.integers(1, 20, (n_cells, 3)).astype(float)

    n_arch = np.zeros((n_cells, probs.n_archetypes(r)))
    for k in range(3):
        n_arch[:, corner_index(r, k)] = counts[:, k]
    pi = counts.sum(0) / counts.sum()
    grid = probs.simplex_grid(r)[:, probs.TO_ORDINAL]
    logL = probs.log_likelihood_ratio(np.maximum(np.eye(3)[grid.argmax(1)], 1e-300), pi)

    tau_m, nu_m = ordinal.mixture_sites(m, v, c, jnp.asarray(n_arch), jnp.asarray(logL))
    tau_c, nu_c = ordinal.sites(m, v, c, *(jnp.asarray(counts[:, k]) for k in range(3)))

    # a constant offset in the log-likelihood drops out of both derivatives
    assert np.allclose(np.asarray(tau_m), np.asarray(tau_c), rtol=1e-9, atol=1e-10)
    assert np.allclose(np.asarray(nu_m), np.asarray(nu_c), rtol=1e-9, atol=1e-10)


def test_uninformative_post_carries_no_information():
    """A post at the centroid must leave the site untouched."""
    r, c = 6, 0.6
    m = jnp.asarray(np.linspace(-2, 2, 33))
    v = jnp.asarray(np.full(33, 0.4))
    grid = probs.simplex_grid(r)
    centroid = int(np.flatnonzero(np.abs(grid - 1 / 3).sum(1) < 1e-12)[0])

    base = np.zeros((33, probs.n_archetypes(r)))
    base[:, corner_index(r, 2)] = 5.0
    plus = base.copy()
    plus[:, centroid] = 40.0                      # 40 posts that say nothing

    pi = np.full(3, 1 / 3)
    logL = probs.log_likelihood_ratio(probs.archetypes(r, floor=0.0), pi)
    args = (jnp.asarray(logL),)
    t0, n0 = ordinal.mixture_sites(m, v, c, jnp.asarray(base), *args)
    t1, n1 = ordinal.mixture_sites(m, v, c, jnp.asarray(plus), *args)

    assert np.allclose(np.asarray(t0), np.asarray(t1), atol=1e-9)
    assert np.allclose(np.asarray(n0), np.asarray(n1), atol=1e-9)


def test_soft_counts_recover_the_moment_identity():
    """(n, s_sum, s2_sum) must give back the expected counts under q."""
    rng = np.random.default_rng(2)
    q = rng.dirichlet(np.ones(3), 500)                    # neutral, favor, against
    neu, fav, ag = q[:, 0], q[:, 1], q[:, 2]
    s_sum = (fav - ag).sum()
    s2_sum = (fav + ag).sum()
    n = float(len(q))

    assert np.isclose((s2_sum + s_sum) / 2, fav.sum())
    assert np.isclose((s2_sum - s_sum) / 2, ag.sum())
    assert np.isclose(n - s2_sum, neu.sum())


# --------------------------------------------------------------- recovery

M_R, J_R, T_R, K_R, C_R, BIN_DAYS = 25, 15, 40, 3, 1.2, 2
STRENGTH = 1.5                 # gives ~0.77 argmax accuracy, near this classifier's


def _synth_cells(path, strength, seed=0):
    """Cells whose labels come from a simulated classifier, not from the truth.

    The classifier observes each true label through Gaussian noise and reports
    the exact posterior under a flat prior, so the probabilities are calibrated
    and the mixture form is the correct likelihood by construction.
    """
    import datetime
    import polars as pl
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    W = rng.normal(size=(J_R, K_R)) * 0.8
    b = rng.normal(size=J_R) * 0.3
    z = np.empty((M_R, T_R, K_R))
    z[:, :, :2] = rng.normal(size=(M_R, 1, 2))
    z[:, :, 2] = np.cumsum(rng.normal(scale=0.15, size=(M_R, T_R)), 1)

    L = probs.n_archetypes(6)
    rows = []
    t0 = datetime.datetime(2020, 1, 1)
    for m in range(M_R):
        rng = np.random.default_rng(1000 + m)
        for j in range(J_R):
            n = rng.poisson(6, size=T_R)
            f = z[m] @ W[j] + b[j]
            p_neg, p_pos = norm.cdf(-C_R - f), norm.cdf(f - C_R)
            for t in np.flatnonzero(n > 0):
                cnt = rng.multinomial(n[t], [p_neg[t], 1 - p_neg[t] - p_pos[t], p_pos[t]])
                y = np.repeat([0, 1, 2], cnt)
                e = rng.normal(size=(len(y), 3)) + strength * np.eye(3)[y]
                w = np.exp(strength * (e - e.max(1, keepdims=True)))
                q_ord = w / w.sum(1, keepdims=True)
                lat = np.bincount(probs.assign(q_ord[:, np.argsort(probs.TO_ORDINAL)], 6),
                                  minlength=L).astype(float)
                s = np.array([-1.0, 0.0, 1.0])[q_ord.argmax(1)]
                rows.append([f'seed{m:03d}', f'target{j:02d}',
                             t0 + datetime.timedelta(days=int(t) * BIN_DAYS),
                             float(s.sum()), float((s ** 2).sum()), int(n[t])] + lat.tolist())
    schema = ['SeedName', 'target', 'bin', 's_sum', 's2_sum', 'n'] + [f'q{i}' for i in range(L)]
    pl.DataFrame(rows, schema=schema, orient='row').write_parquet(path, compression='zstd')
    return z


def _recovery_r2(fitted, true):
    """Mean R^2 of each true dimension on the fitted basis: rotation-invariant."""
    X = np.c_[fitted, np.ones(len(fitted))]
    out = []
    for k in range(true.shape[1]):
        y = true[:, k]
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        out.append(1 - resid.var() / y.var())
    return float(np.mean(out))


@pytest.mark.slow
def test_probabilities_recover_the_latent_better_than_labels():
    import tempfile
    import splits
    from latent_gp import LatentConfig, build_latents, coord_cols

    seeds = [f'seed{m:03d}' for m in range(M_R)]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'cells.parquet.zstd')
        z = _synth_cells(path, STRENGTH)

        got = {}
        for obs_model in ('hard', 'soft', 'mixture'):
            out = build_latents(
                LatentConfig(cells_path=path, n_dims=K_R, n_fast=1, fast_tau=20.0,
                             bin_factor=1, iters=10, infer_iters=6, min_target_volume=0,
                             obs_model=obs_model, prob_resolution=6),
                splits.SplitSpec(holdout_days=30),
                {s: 'train' for s in seeds}, log=lambda *a: None)
            m_i = np.array([seeds.index(s) for s in out['filter_value'].to_list()])
            t_i = ((out['createtime'].to_numpy().astype('datetime64[D]')
                    - np.datetime64('2020-01-01')).astype(int) // BIN_DAYS)
            got[obs_model] = _recovery_r2(np.stack(out[coord_cols(K_R)[0]].to_numpy()),
                                          z[m_i, t_i])

        assert got['mixture'] > got['hard'] + 0.02, got
        assert got['mixture'] > got['soft'], got


# ------------------------------------------------------------ calibration

# The stance finetune's own table, rows = true label, columns = prediction,
# in the classifier's (neutral, favor, against) order.
FINETUNE_COUNTS = [[3710, 577, 475], [723, 4113, 776], [830, 1026, 4742]]


def test_confusion_counts_are_permuted_into_ordinal_order():
    from latent_gp import calibrate

    out = calibrate.from_confusion_counts(FINETUNE_COUNTS)
    assert out['order'] == ['AGAINST', 'NEUTRAL', 'FAVOR']
    assert out['n'] == 16972
    assert np.isclose(out['accuracy'], 12565 / 16972)

    c = np.asarray(out['confusion'])
    assert np.allclose(c.sum(1), 1.0)
    # true AGAINST is row 2 of the source, reordered to (against, neutral, favor)
    assert np.allclose(c[0], np.array([4742, 830, 1026]) / 6598)
    assert np.allclose(c[1], np.array([475, 3710, 577]) / 4762)
    assert np.allclose(c[2], np.array([776, 723, 4113]) / 5612)


def test_channel_log_ratio_reads_the_prediction_column():
    from latent_gp import calibrate

    c = np.asarray(calibrate.from_confusion_counts(FINETUNE_COUNTS)['confusion'])
    logL = calibrate.channel_log_ratio(c)
    for lattice_index, ordinal_class in enumerate(calibrate.corner_order()):
        assert np.allclose(logL[lattice_index], np.log(c[:, ordinal_class]))


def test_channel_with_a_perfect_classifier_is_the_count_likelihood():
    """An identity channel claims no error, so it must reduce to hard labels."""
    from latent_gp import calibrate

    rng = np.random.default_rng(3)
    n_cells, c = 40, 0.65
    m = jnp.asarray(rng.normal(0, 1.0, n_cells))
    v = jnp.asarray(rng.uniform(0.1, 0.7, n_cells))
    counts = rng.integers(1, 15, (n_cells, 3)).astype(float)      # neg, neu, pos

    n_arch = calibrate.corner_counts(counts[:, 0], counts[:, 1], counts[:, 2])
    logL = calibrate.channel_log_ratio(np.eye(3))

    tau_c, nu_c = ordinal.sites(m, v, c, *(jnp.asarray(counts[:, k]) for k in range(3)))
    tau_m, nu_m = ordinal.mixture_sites(m, v, c, jnp.asarray(n_arch), jnp.asarray(logL))

    assert np.allclose(np.asarray(tau_m), np.asarray(tau_c), rtol=1e-9, atol=1e-10)
    assert np.allclose(np.asarray(nu_m), np.asarray(nu_c), rtol=1e-9, atol=1e-10)


def test_a_noisier_channel_shrinks_the_site_precision():
    """More assumed classifier error must mean each post carries less weight."""
    from latent_gp import calibrate

    rng = np.random.default_rng(4)
    n_cells, c = 40, 0.65
    m = jnp.asarray(rng.normal(0, 1.0, n_cells))
    v = jnp.asarray(rng.uniform(0.1, 0.7, n_cells))
    counts = rng.integers(2, 15, (n_cells, 3)).astype(float)
    n_arch = jnp.asarray(calibrate.corner_counts(counts[:, 0], counts[:, 1], counts[:, 2]))

    def precision(accuracy):
        off = (1 - accuracy) / 2
        C = np.full((3, 3), off) + np.eye(3) * (accuracy - off)
        tau, _ = ordinal.mixture_sites(m, v, c, n_arch,
                                       jnp.asarray(calibrate.channel_log_ratio(C)))
        return float(np.asarray(tau).sum())

    assert precision(0.99) > precision(0.85) > precision(0.74) > precision(0.5)


def test_a_saturated_cell_leaves_the_state_alone():
    """Far from the transition, a bounded likelihood must contribute nothing.

    Otherwise the site has a live gradient and no curvature, and the smoother
    divides by that precision.
    """
    from latent_gp import calibrate

    C = calibrate.from_confusion_counts(FINETUNE_COUNTS)['confusion']
    logL = jnp.asarray(calibrate.channel_log_ratio(C))
    n_arch = jnp.asarray(calibrate.corner_counts(*[np.array([20.0])] * 3))

    far = jnp.asarray([60.0])
    v = jnp.asarray([1.0])
    tau, nu = ordinal.mixture_sites(far, v, 0.8, n_arch, logL)
    assert float(tau[0]) == pytest.approx(ordinal.INERT_PRECISION, rel=1e-9)
    assert float(nu[0]) == pytest.approx(float(far[0]), rel=1e-9)

    near = jnp.asarray([0.0])
    tau_near, _ = ordinal.mixture_sites(near, v, 0.8, n_arch, logL)
    assert float(tau_near[0]) > 1e-3, 'a cell in the transition must stay informative'
