"""Model-free test for temporal drift in raw cell statistics.

For each (seed, target) pair, compare cell statistics separated by a lag and
subtract the sampling variance they would show if the underlying rate were
fixed. What survives is real variation:

    lag-independent offset -> within-cell clustering (posts are not iid)
    growth with lag        -> genuine drift, on the timescale where it saturates
    saturation            -> the drift is bounded, i.e. mean-reverting
    unbounded growth      -> a random walk

Makes no assumption about the factor model, the prior, or the likelihood. The
sampling floor is pooled *within-cell* variance, which contains no between-time
variation, so drift cannot leak into the floor. A time-permutation null
calibrates what "no drift" looks like on the same cells.

Two statistics are measured: the cell mean stance (what a Gaussian likelihood
sees) and the neutral share (which it cannot see at all).
"""

import argparse

import numpy as np
import polars as pl

EDGES = [1, 2, 4, 8, 16, 32, 48, 64, 80]

MEAN_STANCE = 'mean stance (what a Gaussian likelihood sees)'
NEUTRAL_SHARE = 'neutral share (invisible to a Gaussian likelihood)'


def bucketed(p, val, val_b, noise, dt, label):
    p = p.with_columns([
        ((pl.col(val) - pl.col(val_b)) ** 2).alias('sq'),
        noise.alias('noise'),
        (1.0 / (1.0 / pl.col('n') + 1.0 / pl.col('n_b'))).alias('w'),
    ])
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        b = p.filter((pl.col('lag') >= lo) & (pl.col('lag') < hi))
        if not len(b):
            continue
        w = b['w'].to_numpy()
        raw = float(np.average(b['sq'].to_numpy(), weights=w))
        noi = float(np.average(b['noise'].to_numpy(), weights=w))
        rows.append((0.5 * (lo + hi) * dt, len(b), raw, noi, raw - noi))
    return rows


def show(name, rows, null_rows):
    """real - null removes composition: long lags only exist for long-span,
    high-volume pairs, which carry less clustering.
    """
    print(f"\n--- {name} ---")
    print(f"{'lag (days)':>11} {'pairs':>9} {'excess':>9} {'null':>9} "
          f"{'real-null':>10}  {'corr':>6}")
    print('-' * 60)
    diffs = [e - n for (*_, e), (*_, n) in zip(rows, null_rows)]
    span = diffs[-1] - diffs[0]
    for (mid, n, raw, noi, exc), d in zip(rows, diffs, strict=True):
        # excess(lag) = clustering + 2 Var(drift) (1 - corr(lag)); the diff
        # spans that from corr~1 at short lag to corr~0 once decorrelated
        corr = 1.0 - (d - diffs[0]) / span if span > 0 else float('nan')
        print(f"{mid:11.0f} {n:9,} {exc:9.5f} {null_rows[0][4]:9.5f} "
              f"{d:10.5f}  {corr:6.3f}")
    print(f"  drift variance spanned (2*Var): {span:.5f}"
          f"   -> drift sd ~ {np.sqrt(max(span, 0) / 2):.4f}")


def curves(df, meta, min_n=5, max_lag=60, log=print):
    """Lag-bucketed excess variance for the real cells and a permuted null.

    Returns {statistic name: {'real': rows, 'null': rows}}, rows as
    `bucketed` produces them: (lag_days, n_pairs, excess, noise, real-noise).
    """
    dt = meta['dt']

    cells = df.filter(pl.col('n') >= min_n).with_columns([
        (pl.col('s_sum') / pl.col('n')).alias('ybar'),
        (pl.col('n_neu') / pl.col('n')).alias('pneu'),
        # within-cell scatter only -- no between-time component
        (pl.col('s2_sum') - pl.col('s_sum') ** 2 / pl.col('n')).alias('ss_w'),
    ])
    pair = cells.group_by(['m', 'j']).agg(
        pl.col('ss_w').sum().alias('SSW'), (pl.col('n') - 1).sum().alias('DFW'),
        pl.col('n_neu').sum().alias('NEU'), pl.col('n').sum().alias('N'),
        pl.len().alias('n_bins'))
    pair = pair.filter((pl.col('n_bins') >= 3) & (pl.col('DFW') > 0)).with_columns([
        (pl.col('SSW') / pl.col('DFW')).alias('v'),          # pooled within-cell var
        (pl.col('NEU') / pl.col('N')).alias('p'),            # pooled neutral share
    ])
    cells = cells.join(pair.select(['m', 'j', 'v', 'p']), on=['m', 'j'], how='inner')
    log(f"cells n>={min_n} in pairs with >=3 bins: {len(cells)} "
        f"over {len(pair)} (seed,target) pairs")

    # Null: permute the cell values within each (seed, target) while holding the
    # time grid fixed. This preserves every lag, weight and noise term exactly
    # and destroys only the association between time and value.
    cells = cells.sort(['m', 'j', 't'])
    grp = cells.select(['m', 'j']).to_numpy()
    starts = np.flatnonzero(np.r_[True, (grp[1:] != grp[:-1]).any(1)])
    bounds = np.r_[starts, len(cells)]
    rng = np.random.default_rng(0)
    order = np.arange(len(cells))
    for a, b in zip(bounds[:-1], bounds[1:]):
        order[a:b] = a + rng.permutation(b - a)
    moved = ['ybar', 'pneu', 'n', 'v', 'p']
    perm = cells.with_columns(
        [pl.col(c).gather(order).alias(c) for c in moved])

    out = {}
    for tag, src in (('real', cells), ('null', perm)):
        p = src.join(src, on=['m', 'j'], suffix='_b')
        p = p.filter(pl.col('t_b') > pl.col('t')).with_columns(
            (pl.col('t_b') - pl.col('t')).alias('lag')).filter(pl.col('lag') <= max_lag)
        if tag == 'real':
            log(f"lagged cell pairs: {len(p):,}")
        noise_y = pl.col('v') * (1.0 / pl.col('n') + 1.0 / pl.col('n_b'))
        noise_p = (pl.col('p') * (1 - pl.col('p'))
                   * (1.0 / pl.col('n') + 1.0 / pl.col('n_b')))
        out[tag] = {
            MEAN_STANCE: bucketed(p, 'ybar', 'ybar_b', noise_y, dt, tag),
            NEUTRAL_SHARE: bucketed(p, 'pneu', 'pneu_b', noise_p, dt, tag),
        }

    return {name: {'real': out['real'][name], 'null': out['null'][name]}
            for name in out['real']}


def verdict(rows, null_rows, tail_frac=0.34):
    """Does the drift variance saturate with lag, or keep growing?

    `real - null` is clustering + 2 Var(drift) (1 - corr(lag)); the clustering
    term is lag-independent, so its growth from the shortest lag is what the
    drift contributes. A plateau means the drift is bounded (mean-reverting);
    growth sustained into the longest lags is what a random walk looks like.
    """
    d = np.array([e - n for (*_, e), (*_, n) in zip(rows, null_rows, strict=True)])
    lags = np.array([r[0] for r in rows])
    span = float(d[-1] - d[0])
    n_tail = max(2, int(round(len(d) * tail_frac)))
    tail_span = float(d[-1] - d[-n_tail])
    # share of the total rise still being accumulated over the last lags
    tail_share = tail_span / span if span > 0 else float('nan')
    return {
        'lags_days': lags,
        'excess_over_null': d,
        'span': span,
        'drift_sd': float(np.sqrt(max(span, 0.0) / 2)),
        'tail_share': tail_share,
        'tail_lags_days': (float(lags[-n_tail]), float(lags[-1])),
        'saturates': bool(span > 0 and tail_share < 0.1),
    }


def main():
    # deferred: `curves` and `verdict` are plain numpy, and loading the cells
    # is the only part of this module that needs jax
    from . import cells as prep

    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--bin-factor', type=int, default=8)
    ap.add_argument('--min-n', type=int, default=5)
    ap.add_argument('--max-lag', type=int, default=60)
    args = ap.parse_args()

    df, meta = prep.load(args.data, args.bin_factor)
    out = curves(df, meta, args.min_n, args.max_lag)
    for name, got in out.items():
        show(name, got['real'], got['null'])


if __name__ == '__main__':
    main()
