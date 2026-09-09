"""Ordered-probit cell likelihood and its Gaussian site approximation.

Every post in a cell shares the same predictor f, so under labels the exact
cell likelihood depends on the data only through the three category counts --
the same size as the Gaussian sufficient statistics. Thresholds are symmetric
(-c, +c); a global shift is absorbed by the per-target intercept b_j and a
global scale by W.

That count likelihood is log-concave in f, so its site precision is always
positive. The mixture form at the bottom of the file, which consumes the
classifier's per-post probabilities instead of its labels, is not, and relies
on the precision floor.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from . import _jax  # noqa: F401  -- x64 before the first array

GH_N = 12
_x, _w = np.polynomial.hermite_e.hermegauss(GH_N)
GH_X = jnp.asarray(_x)
GH_W = jnp.asarray(_w / np.sqrt(2 * np.pi))


def _log_diff_ndtr(a, b):
    """log(Phi(b) - Phi(a)) for b > a, evaluated in whichever tail is stable."""
    swap = (a + b) > 0
    lo = jnp.where(swap, -b, a)
    hi = jnp.where(swap, -a, b)
    l_lo, l_hi = log_ndtr(lo), log_ndtr(hi)
    return l_hi + jnp.log1p(-jnp.exp(jnp.minimum(l_lo - l_hi, -1e-10)))


def log_probs(f, c):
    """Log P(-1|f), P(0|f), P(+1|f)."""
    return (log_ndtr(-c - f),
            _log_diff_ndtr(-c - f, c - f),
            log_ndtr(f - c))


def cell_loglik(f, c, n_neg, n_neu, n_pos):
    lneg, lneu, lpos = log_probs(f, c)
    return n_neg * lneg + n_neu * lneu + n_pos * lpos


def _quad_nodes(m, v):
    return m[..., None] + jnp.sqrt(jnp.maximum(v, 1e-12))[..., None] * GH_X


def expected_loglik(m, v, c, n_neg, n_neu, n_pos):
    """E[log p(cell | f)] under the current Gaussian marginal f ~ N(m, v)."""
    ll = cell_loglik(_quad_nodes(m, v), c,
                     n_neg[..., None], n_neu[..., None], n_pos[..., None])
    return (GH_W * ll).sum(-1)


def _total(m, v, c, nn, nu_, np_):
    return expected_loglik(m, v, c, nn, nu_, np_).sum()


_d1 = jax.grad(_total, argnums=0)
_d2 = jax.grad(lambda *a: _d1(*a).sum(), argnums=0)
_dc = jax.grad(_total, argnums=2)
_dcc = jax.grad(_dc, argnums=2)


@jax.jit
def sites(m, v, c, n_neg, n_neu, n_pos):
    """Gaussian site (precision, pseudo-observation) matching E[log p] to 2nd order.

    This is the variational (CVI) update: derivatives are taken of the expected
    log-likelihood under the current marginal, not of the likelihood at a point.
    """
    g1 = _d1(m, v, c, n_neg, n_neu, n_pos)
    g2 = _d2(m, v, c, n_neg, n_neu, n_pos)
    tau = jnp.maximum(-g2, 1e-8)
    return tau, m + g1 / tau


@jax.jit
def _threshold_derivs(m, v, c, n_neg, n_neu, n_pos):
    return (_dc(m, v, c, n_neg, n_neu, n_pos),
            _dcc(m, v, c, n_neg, n_neu, n_pos))


def newton_threshold(m, v, c, n_neg, n_neu, n_pos, chunk=750_000):
    """One damped Newton step on the shared threshold.

    The objective sums over every cell, so the derivatives are accumulated
    chunk by chunk rather than materialising the quadrature for all of them.
    """
    g = h = 0.0
    for i in range(0, m.shape[0], chunk):
        s = slice(i, i + chunk)
        gi, hi = _threshold_derivs(m[s], v[s], c, n_neg[s], n_neu[s], n_pos[s])
        g += float(gi); h += float(hi)
    step = -g / h if h < -1e-12 else 0.0
    return float(np.clip(c + np.clip(step, -0.25, 0.25), 0.05, 5.0))


@jax.jit
def predictive(m, v, c):
    """Posterior-averaged category probabilities for one cell."""
    lp = log_probs(_quad_nodes(m, v), c)
    return tuple((GH_W * jnp.exp(l)).sum(-1) for l in lp)


def _chunked(fn, n, chunk, *arrs, scalars=()):
    """Apply a jitted per-cell function in slices; quadrature is memory-hungry."""
    if n <= chunk:
        return fn(*arrs, *scalars)
    outs = [fn(*(a[i:i + chunk] for a in arrs), *scalars)
            for i in range(0, n, chunk)]
    return tuple(jnp.concatenate([o[k] for o in outs]) for k in range(len(outs[0])))


def sites_chunked(m, v, c, n_neg, n_neu, n_pos, chunk=750_000):
    return _chunked(lambda *a: sites(a[0], a[1], c, a[2], a[3], a[4]),
                    m.shape[0], chunk, m, v, n_neg, n_neu, n_pos)


def predictive_chunked(m, v, c, chunk=750_000):
    return _chunked(lambda *a: predictive(a[0], a[1], c), m.shape[0], chunk, m, v)


def init_threshold(n_neg, n_neu, n_pos):
    """Match the marginal neutral share at f = 0."""
    from scipy.stats import norm
    share = float(n_neu.sum() / (n_neg.sum() + n_neu.sum() + n_pos.sum()))
    return float(norm.ppf(0.5 + share / 2))


# ------------------------------------------------ soft evidence (mixture form)
#
# With per-post classifier probabilities the cell no longer reduces to three
# counts, so posts are binned onto a lattice over the simplex and the cell
# carries one count per lattice point l with log-likelihood-ratio row logL[l].
#
# log p(cell | f) = sum_l n_l * log sum_k exp(logL[l,k]) P(k | f)
#
# This form is not log-concave in f: a post that rules out the middle category
# but splits between the ends gives P(-1|f) + P(+1|f), which is convex. It is
# also bounded, so once the classifier's error rate can explain a cell outright
# the likelihood goes flat and the cell stops saying anything about f.
#
# A flat cell has to contribute nothing, not a little. The smoother works in
# information form and recovers the mean by dividing by the precision, so a
# site with a live gradient and near-zero curvature is an unbounded
# pseudo-observation, and one iteration of those sends the latent scale to
# hundreds while W collapses to compensate.


def mixture_expected_loglik(m, v, c, n_arch, logL):
    """E[log p(cell | f)] under f ~ N(m, v), summed over lattice points."""
    lp = jnp.stack(log_probs(_quad_nodes(m, v), c), -1)         # (cells, GH, 3)
    per = logsumexp(logL + lp[..., None, :], axis=-1)           # (cells, GH, L)
    return (GH_W * (n_arch[:, None, :] * per).sum(-1)).sum(-1)


def _mix_total(m, v, c, n_arch, logL):
    return mixture_expected_loglik(m, v, c, n_arch, logL).sum()


_m_d1 = jax.grad(_mix_total, argnums=0)
_m_d2 = jax.grad(lambda *a: _m_d1(*a).sum(), argnums=0)
_m_dc = jax.grad(_mix_total, argnums=2)
_m_dcc = jax.grad(_m_dc, argnums=2)


# Small enough to be negligible against a prior precision of order one, so a
# site pinned here is inert rather than merely weak.
INERT_PRECISION = 1e-6


@jax.jit
def mixture_sites(m, v, c, n_arch, logL):
    """Gaussian site, with cells whose likelihood has gone flat left inert."""
    g1 = _m_d1(m, v, c, n_arch, logL)
    g2 = _m_d2(m, v, c, n_arch, logL)
    informative = g2 < -INERT_PRECISION
    tau = jnp.where(informative, -g2, INERT_PRECISION)
    return tau, jnp.where(informative, m + g1 / tau, m)


@jax.jit
def _mixture_threshold_derivs(m, v, c, n_arch, logL):
    return (_m_dc(m, v, c, n_arch, logL), _m_dcc(m, v, c, n_arch, logL))


def mixture_chunk(n_arch_cols, budget=750_000):
    """Cells per slice: the quadrature is materialised per lattice point."""
    return max(budget // max(int(n_arch_cols), 1), 25_000)


# ----------------------------------------------------- observation model

class Counts:
    """Three-count observation.

    Integer counts under hard labels, expected counts under the classifier
    posterior; the arithmetic is identical either way.
    """

    def __init__(self, d):
        self.args = (d['n_neg'], d['n_neu'], d['n_pos'])

    def init_threshold(self):
        return init_threshold(*self.args)

    def sites(self, m, v, c):
        return sites_chunked(m, v, c, *self.args)

    def threshold_step(self, m, v, c):
        return newton_threshold(m, v, c, *self.args)


class Mixture:
    """Lattice-count observation using the per-post probabilities in full."""

    def __init__(self, d, logL):
        self.n_arch = np.asarray(d['n_arch'])
        self.logL = jnp.asarray(logL)
        self.counts = (d['n_neg'], d['n_neu'], d['n_pos'])
        self.chunk = mixture_chunk(self.logL.shape[0])

    def init_threshold(self):
        # the same marginal-share heuristic; the exact soft counts are kept on
        # the packed cell alongside the lattice counts
        return init_threshold(*self.counts)

    def sites(self, m, v, c):
        outs = [mixture_sites(m[s], v[s], c, jnp.asarray(self.n_arch[s]), self.logL)
                for s in self._slices(m.shape[0])]
        return (jnp.concatenate([o[0] for o in outs]),
                jnp.concatenate([o[1] for o in outs]))

    def threshold_step(self, m, v, c):
        g = h = 0.0
        for s in self._slices(m.shape[0]):
            gi, hi = _mixture_threshold_derivs(m[s], v[s], c,
                                               jnp.asarray(self.n_arch[s]), self.logL)
            g += float(gi); h += float(hi)
        step = -g / h if h < -1e-12 else 0.0
        return float(np.clip(c + np.clip(step, -0.25, 0.25), 0.05, 5.0))

    def _slices(self, n):
        return [slice(i, i + self.chunk) for i in range(0, n, self.chunk)]


def observation(d, logL=None):
    """Pick the observation model from what the packed cell carries."""
    if logL is None:
        return Counts(d)
    if 'n_arch' not in d:
        raise ValueError('mixture observation needs lattice counts: rebuild the '
                         'aggregate with probabilities')
    return Mixture(d, logL)
