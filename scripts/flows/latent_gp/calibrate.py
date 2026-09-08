"""Calibrate the stance classifier against manually coded pairs.

Two artefacts the latent-GP fit can consume, both in ordinal order
(against, neutral, favor):

  temperature   one parameter, T > 1 flattening an overconfident posterior.
                Cheap in gold labels, and it keeps the per-post information the
                probabilities carry.
  confusion     P(classifier label | coded label), the classic
                misclassification channel. Uses labels only, so it says nothing
                about which posts were hard, but it is what a few hundred coded
                pairs can actually support.

score_coded_posts.py is the report -- accuracy, per-class F1, agreement,
reliability. This module writes only what the model reads.
"""

import argparse
import glob
import json
import os

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from . import probs

ORDINAL = list(probs.ORDINAL_ORDER)                 # AGAINST, NEUTRAL, FAVOR


def _gold_index(labels):
    lookup = {s: i for i, s in enumerate(ORDINAL)}
    return np.array([lookup[str(x).strip().upper()] for x in labels])


def fit_temperature(q, gold, bounds=(0.2, 10.0)):
    """Temperature minimising the coded pairs' negative log-likelihood."""
    y = _gold_index(gold)
    rows = np.arange(len(y))

    def nll(log_t):
        p = probs.temper(q, float(np.exp(log_t)))
        return -np.log(np.maximum(p[rows, y], 1e-300)).mean()

    r = minimize_scalar(nll, bounds=tuple(np.log(bounds)), method='bounded')
    return float(np.exp(r.x)), float(nll(r.x)), float(nll(0.0))


def confusion(q, gold, smoothing=1.0):
    """P(classifier label | coded label), rows summing to one.

    Smoothed because a 3x3 table from a few hundred pairs has cells that are
    empty by luck, and a zero there would tell the model that confusion is
    impossible rather than merely unobserved.
    """
    y = _gold_index(gold)
    pred = np.asarray(q).argmax(1)
    tab = np.full((3, 3), float(smoothing))
    np.add.at(tab, (y, pred), 1.0)
    return tab / tab.sum(1, keepdims=True)


def channel_log_ratio(C):
    """log-likelihood-ratio rows for the three corner archetypes of a lattice.

    Archetype l is 'the classifier said class o(l)', whose likelihood in f is
    sum_k P(k | f) C[k, o]. The marginal that would normalise it is constant
    per archetype and so drops out of every derivative.
    """
    C = np.asarray(C, dtype=np.float64)
    order = corner_order()
    return np.log(np.maximum(C[:, order], 1e-300)).T


def corner_order():
    """Ordinal class asserted by each corner of the resolution-1 lattice."""
    return probs.simplex_grid(1)[:, probs.TO_ORDINAL].argmax(1)


def corner_counts(n_neg, n_neu, n_pos):
    """Category counts placed on the resolution-1 lattice's three corners."""
    counts = np.stack([n_neg, n_neu, n_pos], 1)
    out = np.zeros_like(counts)
    out[:, np.arange(3)] = counts[:, corner_order()]
    return out


def report(q, gold):
    T, nll_T, nll_1 = fit_temperature(q, gold)
    y = _gold_index(gold)
    pred = np.asarray(q).argmax(1)
    return {
        'n': int(len(y)),
        'accuracy': float((pred == y).mean()),
        'temperature': T,
        'nll_calibrated': nll_T,
        'nll_raw': nll_1,
        'confusion': confusion(q, gold).tolist(),
        'order': ORDINAL,
    }


# ------------------------------------------------------------------- CLI

def load_coded(coded_paths, probs_dir):
    """Coded pairs joined to the run's probabilities, in ordinal order.

    Joins on (post, platform, target). Target text is rewritten whenever
    extraction changes, so a coding round predating the current run will mostly
    fail to join -- the count is reported rather than assumed.
    """
    coded = pl.concat([pl.read_csv(p, infer_schema_length=10_000)
                       for p in coded_paths], how='diagonal_relaxed')
    coded = coded.filter(pl.col('coded_stance').is_not_null())
    if 'coded_relevant' in coded.columns:
        coded = coded.filter(pl.col('coded_relevant') == 1)
    keys = set(zip(coded['post_id'].cast(str).to_list(), coded['platform'].to_list()))

    parts = []
    for f in sorted(glob.glob(os.path.join(probs_dir, '*stance_probs*'))):
        d = pl.read_parquet(f).explode(['Targets', 'Probs']).drop_nulls('Targets')
        d = d.filter(pl.struct('id', 'platform').map_elements(
            lambda s: (s['id'], s['platform']) in keys, return_dtype=pl.Boolean))
        if len(d):
            parts.append(d)
    if not parts:
        return coded.head(0), np.zeros((0, 3))

    run = pl.concat(parts).unique(['id', 'platform', 'Targets'])
    j = coded.with_columns(pl.col('post_id').cast(str)).join(
        run, left_on=['post_id', 'platform', 'target'],
        right_on=['id', 'platform', 'Targets'], how='inner')
    q_cls = np.stack(j['Probs'].to_list()) if len(j) else np.zeros((0, 3))
    return j, q_cls[:, probs.TO_ORDINAL] if len(j) else q_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--coded', action='append', required=True,
                    help='CSV exported by the coding page')
    ap.add_argument('--probs-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-pairs', type=int, default=100)
    args = ap.parse_args()

    j, q = load_coded(args.coded, args.probs_dir)
    print(f'coded pairs joined to the run: {len(j)}')
    if len(j) < args.min_pairs:
        raise SystemExit(
            f'only {len(j)} pairs joined, below --min-pairs {args.min_pairs}. Target '
            f'text is rewritten when extraction changes, so a coding round drawn '
            f'from an earlier run will not join; draw a fresh sample with '
            f'sample_classified_posts.py and code that.')

    out = report(q, j['coded_stance'].to_list())
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
