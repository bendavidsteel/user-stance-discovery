"""Prior components: what each kernel implies for the latent's covariance.

The fit only ever sees a component through the (F, Q, P0) it contributes, so
these check the implied prior covariance rather than the blocks themselves.
"""

import numpy as np
import pytest

from . import core as gpfa
from . import fit

DT = 16.0

# the jitter build_ssm adds to Q accumulates over the horizon, and an additive
# form carries more states than the single component it equals, so two exactly
# equal kernels still differ by a little more than machine epsilon
JITTER_FLOOR = 1e-9


def prior_cov(comps, K=2, T=40, dt=DT):
    """Cov(f_s, f_t) for one latent dimension under a component list."""
    F, Q, P0, S = gpfa.build_ssm(dt, comps, K)
    P = [np.asarray(P0)]
    for _ in range(T - 1):
        P.append(F @ P[-1] @ F.T + Q)
    Fp = [np.eye(F.shape[0])]
    for _ in range(T - 1):
        Fp.append(F @ Fp[-1])
    return np.array([[float((S @ Fp[t - s] @ P[s] @ S.T)[0, 0]) if t >= s else 0.0
                      for s in range(T)] for t in range(T)])


def variance(comps, **kw):
    return np.diag(prior_cov(comps, **kw))


# ---------------------------------------------------------------- degeneracy

def test_two_wieners_are_one_wiener():
    """Summing Wiener components on one dimension buys nothing.

    Both have F = 1, so the sum of the two states is itself a random walk whose
    Q is the sum of theirs -- a single Wiener at the combined timescale.
    """
    two = prior_cov([dict(kind='wiener', tau=640., p0=0.5),
                     dict(kind='wiener', tau=640., p0=0.5)])
    one = prior_cov([dict(kind='wiener', tau=320., p0=1.0)])
    assert np.abs(two - one).max() / np.abs(one).max() < JITTER_FLOOR


def test_const_plus_wiener_is_one_wiener():
    """A frozen level only shifts the Wiener's initial variance."""
    add = prior_cov([dict(kind='const', var=1.0),
                     dict(kind='wiener', tau=640., p0=0.0)])
    one = prior_cov([dict(kind='wiener', tau=640., p0=1.0)])
    assert np.abs(add - one).max() / np.abs(one).max() < JITTER_FLOOR


def test_wiener_plus_matern_is_not_degenerate():
    """A real two-timescale sum needs components of different shape."""
    add = prior_cov([dict(kind='wiener', tau=2560., p0=1.0),
                     dict(kind='matern32', tau=80., var=1.0)])
    one = prior_cov([dict(kind='wiener', tau=2560., p0=2.0)])
    assert np.abs(add - one).max() / np.abs(one).max() > 1e-3


# ------------------------------------------------------------------------ ou

def test_ou_is_stationary():
    v = variance([dict(kind='ou', tau=640., var=1.0)])
    assert np.abs(v - 1.0).max() < 1e-10


def test_wiener_is_not_stationary():
    v = variance([dict(kind='wiener', tau=640., var=1.0)])
    assert v[-1] > 1.5 * v[0]


@pytest.mark.parametrize('tau', [640., 2560.])
def test_ou_matches_wiener_increments_at_short_lag(tau):
    """The tau scaling is chosen so the two differ only by reverting."""
    ou = prior_cov([dict(kind='ou', tau=tau, var=1.0)])
    wi = prior_cov([dict(kind='wiener', tau=tau, var=1.0, p0=1.0)])

    def first_increment(C):
        return C[0, 0] + C[1, 1] - 2 * C[1, 0]

    a, b = first_increment(ou), first_increment(wi)
    assert abs(a - b) / b < 2.0 * DT / tau


def test_ou_reverts_where_wiener_does_not():
    """Past its correlation time the OU forgets where it started; the Wiener does not."""
    def lag_corr(comps, lag=60):
        C = prior_cov(comps, T=lag + 1)
        return C[lag, 0] / np.sqrt(C[0, 0] * C[lag, lag])

    # lag 60 bins at dt=16 is 960 days, three OU correlation times at this tau
    assert lag_corr([dict(kind='ou', tau=160., var=1.0)]) < 0.15
    assert lag_corr([dict(kind='wiener', tau=160., var=1.0, p0=1.0)]) > 0.3


def test_ou_becomes_const_as_tau_grows():
    ou = prior_cov([dict(kind='ou', tau=1e7, var=1.0)])
    const = prior_cov([dict(kind='const', var=1.0)])
    assert np.abs(ou - const).max() < 1e-4


def test_ou_is_rougher_than_matern_at_one_tau():
    """Both revert; only the Matern has differentiable paths, so it steps less."""
    def first_increment(comps):
        C = prior_cov(comps)
        return C[0, 0] + C[1, 1] - 2 * C[1, 0]

    assert (first_increment([dict(kind='ou', tau=640., var=1.0)])
            > first_increment([dict(kind='matern32', tau=640., var=1.0)]))


# ------------------------------------------------------- prior_components

def test_homogeneous_mix_is_shared_across_dimensions():
    """n_fast >= K is how the sweep reaches a homogeneous prior."""
    comps = fit.prior_components(K=6, n_fast=6, fast_tau=80., fast_kind='ou')
    assert comps == [dict(kind='ou', tau=80.0, var=1.0)]
    assert not gpfa.is_heterogeneous(comps)


def test_a_real_mix_is_per_dimension_and_fixes_the_basis():
    comps = fit.prior_components(K=3, n_fast=1, fast_tau=80., fast_kind='ou',
                                 slow_kind='const')
    assert len(comps) == 3 and comps[0][0]['kind'] == 'ou'
    assert comps[1][0]['kind'] == 'const'
    assert gpfa.is_heterogeneous(comps)


def test_both_kinds_reach_the_drifting_dimensions():
    for kind in ('wiener', 'ou'):
        comps = fit.prior_components(K=3, n_fast=2, fast_tau=40., fast_kind=kind,
                                     slow_kind='wiener', slow_tau=2560.)
        assert comps[0][0] == dict(kind=kind, tau=40.0, var=1.0)
        assert comps[2][0] == dict(kind='wiener', tau=2560.0, var=1.0)


def test_slow_kind_takes_an_ou():
    comps = fit.prior_components(K=3, n_fast=1, fast_tau=80., slow_kind='ou',
                                 slow_tau=640.)
    assert comps[1][0] == dict(kind='ou', tau=640.0, var=1.0)
