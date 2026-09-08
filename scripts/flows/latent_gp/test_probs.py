"""The soft-evidence path must reduce to the count path on confident data."""

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
