"""Checks on the soft-evidence observation models.

The lattice site is pinned against the count site: an identity channel claims
no classifier error, so it has to reduce exactly to the counts 'hard' would
have used, and a noisier one has to carry less weight per post.
"""

import sys
import os

import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_gp import ordinal, probs


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
