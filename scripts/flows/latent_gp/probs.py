"""Classifier probabilities as soft evidence for the cell likelihood.

The classifier emits a posterior over three classes per post-target pair, not a
label. Three ways to use it, in increasing fidelity:

  hard      the argmax, i.e. the integer counts the model started with
  soft      expected counts, sum_i q_ik -- but an ambiguous post then counts as
            evidence for the middle category, which is not what it is
  mixture   log sum_k (q_ik / pi_k) P(k | f) per post, so a post that says
            nothing contributes a constant, and hence contributes nothing

`mixture` is the likelihood implied by conditioning on the post text rather
than on a label: q_ik / pi_k is p(text | k) / p(text), which leaves the ordered
probit as the only prior over classes. All three agree when the classifier is
confident, so they can be compared on the same aggregate.

Unlike the three-count forms, the mixture does not reduce to a fixed-size
sufficient statistic, so posts are binned onto a lattice over the simplex and a
cell carries one count per lattice point.
"""

import numpy as np

# The classifier writes its probabilities in this order; the ordered probit
# needs them ordered by stance. Verified against the argmax of the saved
# probabilities matching the saved label exactly.
CLASSIFIER_ORDER = ('NEUTRAL', 'FAVOR', 'AGAINST')
ORDINAL_ORDER = ('AGAINST', 'NEUTRAL', 'FAVOR')
TO_ORDINAL = [CLASSIFIER_ORDER.index(s) for s in ORDINAL_ORDER]

STANCE_VALUE = {'AGAINST': -1.0, 'NEUTRAL': 0.0, 'FAVOR': 1.0}


def temper(q, temperature):
    """Raise probabilities to 1/T and renormalise.

    For a softmax classifier this is exactly temperature scaling of the logits,
    so it needs the probabilities only. T > 1 flattens, T < 1 sharpens.
    """
    if temperature == 1.0:
        return np.asarray(q, dtype=np.float64)
    p = np.power(np.maximum(np.asarray(q, dtype=np.float64), 0.0), 1.0 / temperature)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-300)


def simplex_grid(resolution):
    """Lattice points on the 2-simplex at spacing 1/resolution, in a fixed order.

    Use a resolution divisible by 3, or the uniform distribution is not a
    lattice point and the posts carrying no information -- the ones the mixture
    form exists to handle -- get snapped to a point that leans somewhere.
    """
    r = int(resolution)
    pts = [(a, b, r - a - b) for a in range(r + 1) for b in range(r + 1 - a)]
    return np.asarray(pts, dtype=np.float64) / r


def n_archetypes(resolution):
    r = int(resolution)
    return (r + 1) * (r + 2) // 2


def assign(q, resolution, chunk=1_000_000):
    """Index of the nearest lattice point for each row of `q`.

    Nearest in Euclidean distance on the simplex, which for a corner means the
    post was confident enough to be treated as labelled.
    """
    grid = simplex_grid(resolution)
    q = np.asarray(q, dtype=np.float64)
    out = np.empty(len(q), dtype=np.int32)
    for i in range(0, len(q), chunk):
        d = ((q[i:i + chunk, None, :] - grid[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def archetypes(resolution, temperature=1.0, floor=0.01):
    """Lattice points as category distributions in ordinal order.

    `floor` mixes in the uniform distribution. A bare lattice corner asserts
    that a class is impossible, which no classifier's calibration supports, and
    it sends the likelihood ratio of that class to zero.
    """
    q = temper(simplex_grid(resolution), temperature)[:, TO_ORDINAL]
    return (1.0 - floor) * q + floor / 3.0


def log_likelihood_ratio(q_arch, pi):
    """log(q / pi): the log of p(text | k) / p(text), up to a constant.

    Dividing out the classifier's own marginal leaves the ordered probit as the
    only prior over classes, so the per-target intercept keeps meaning what it
    meant under hard labels.
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    return np.log(np.maximum(q_arch, 1e-300)) - np.log(np.maximum(pi, 1e-300))


def marginal(n_neg, n_neu, n_pos):
    """The classifier's implied class marginal, from the soft counts."""
    tot = np.array([np.sum(n_neg), np.sum(n_neu), np.sum(n_pos)], dtype=np.float64)
    return tot / tot.sum()
