"""What an aggregate contains and how big a fit it makes.

Worth re-running whenever the aggregate is rebuilt: it reports how much of the
classifier's output is actually uncertain, which decides whether the soft
observation models have anything to work with, and the (M, J, T) the fit will
see at a given grid, which decides whether it fits in memory.

    python -m latent_gp.describe tmp/gpfa_cells_2022_onwards.parquet.zstd
"""

import argparse

import numpy as np
import polars as pl

from . import cells, probs


def confidence_profile(path):
    """Share of post-target pairs by how confident the classifier was."""
    lf = pl.scan_parquet(path)
    names = lf.collect_schema().names()
    res = cells.stored_resolution(names)
    if res is None:
        print('no lattice counts: aggregate was built from labels alone')
        return
    cols = cells.arch_cols(names)
    tot = lf.select([pl.col(c).sum() for c in cols]).collect().to_numpy()[0]
    grid = probs.simplex_grid(res)
    share = tot / tot.sum()
    conf = grid.max(1)

    print(f'lattice resolution {res} ({len(cols)} points), {tot.sum():,.0f} pairs')
    print('share of pairs by the max probability of the point they landed on:')
    for lo, hi in [(1.0, 1.01), (0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.0, 0.4)]:
        sel = (conf >= lo) & (conf < hi)
        print(f'  [{lo:.1f}, {hi:.2f}): {share[sel].sum():6.1%}')
    cent = np.abs(grid - 1 / 3).sum(1) < 1e-12
    print(f'  exactly uninformative: {share[cent].sum():.2%}')
    q = probs.archetypes(res, floor=0.0)
    pi = (tot @ q) / tot.sum()
    print(f'classifier marginal (against, neutral, favor): {np.round(pi, 4)}')


def target_filter(path):
    lf = pl.scan_parquet(path)
    v = lf.group_by('target').agg(pl.col('n').sum().alias('v')).collect()
    total = v['v'].sum()
    print('\ntargets surviving min_target_volume:')
    for thr in (100, 400, 1000):
        k = v.filter(pl.col('v') >= thr)
        print(f'  >= {thr:5d}: {len(k):>9,} targets, {k["v"].sum():>13,.0f} pairs '
              f'({k["v"].sum() / total:.1%})')


def fit_size(path, bin_factor, min_target_volume, holdouts=(91, 182, 365)):
    df, meta = cells.load(path, bin_factor, min_target_volume=min_target_volume)
    print(f'\nfit at bin_factor={bin_factor} ({meta["dt"]:.0f}-day grid), '
          f'min_target_volume={min_target_volume}:')
    print(f'  M={meta["M"]}  J={meta["J"]}  T={meta["T"]}  cells={len(df):,}')
    print(f'  span {meta["t0"]:%Y-%m-%d} .. {cells.bin_times(meta)[-1]:%Y-%m-%d}')
    t = df['t'].to_numpy()
    for hold in holdouts:
        t_cut, cutoff = cells.cutoff_bin(meta, hold)
        if t_cut >= meta['T']:
            print(f'  holdout {hold:3d}d: cutoff falls past the data')
            continue
        n_in = int((t < t_cut).sum())
        print(f'  holdout {hold:3d}d: cutoff {cutoff:%Y-%m-%d} '
              f'(bin {t_cut}/{meta["T"]}), in-time cells {n_in:,} '
              f'({n_in / len(df):.1%})')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path')
    ap.add_argument('--bin-factor', type=int, default=4)
    ap.add_argument('--min-target-volume', type=int, default=400)
    ap.add_argument('--skip-fit', action='store_true',
                    help='confidence profile only; the fit sizing loads everything')
    args = ap.parse_args()

    print(args.path)
    confidence_profile(args.path)
    target_filter(args.path)
    if not args.skip_fit:
        fit_size(args.path, args.bin_factor, args.min_target_volume)


if __name__ == '__main__':
    main()
