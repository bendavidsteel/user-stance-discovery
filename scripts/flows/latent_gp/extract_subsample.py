"""Subsample seeds from the cell-level stance aggregate.

The aggregate is built by aggregate.py and holds every seed that passes the
actor-type filter. Seeds are taken by systematic sampling on volume rank, which
keeps the sample representative of the whole volume range rather than of its
head.

Run on prometheus.
"""

import argparse
import os

import polars as pl

from . import aggregate

STANCE_DIR = 'data/stance_targets/2022-01-01-onwards_noun_phrase_stance'
PROBS_DIR = STANCE_DIR + '_probs'
PARTS_DIR = 'tmp/gpfa_cell_parts'
CACHE = 'tmp/gpfa_cells_2022_onwards.parquet.zstd'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fraction', type=float, default=0.25)
    ap.add_argument('--min-target-volume', type=int, default=400)
    ap.add_argument('--resolution', type=int, default=6)
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    aggregate.build(STANCE_DIR, PROBS_DIR, PARTS_DIR, args.cache, args.resolution)
    lf = pl.scan_parquet(args.cache)

    keep = (lf.group_by('target').agg(pl.col('n').sum().alias('v'))
              .filter(pl.col('v') >= args.min_target_volume).select('target').collect())
    print(f'targets with volume >= {args.min_target_volume}: {len(keep)}', flush=True)
    lf = lf.join(keep.lazy(), on='target', how='inner')

    seed_vol = lf.group_by('SeedName').agg(pl.col('n').sum().alias('v')) \
                 .sort('v', descending=True).collect()
    step = max(int(round(1.0 / args.fraction)), 1)
    chosen = seed_vol[::step]          # systematic on volume rank
    print(f'seeds: {len(seed_vol)} available -> {len(chosen)} chosen (every {step}th '
          f'by volume rank)', flush=True)
    print(chosen.select(pl.col('v').min().alias('vol_min'),
                        pl.col('v').median().alias('vol_med'),
                        pl.col('v').max().alias('vol_max')), flush=True)

    out = lf.join(chosen.select('SeedName').lazy(), on='SeedName', how='inner').collect()
    print(f'rows {len(out):,}  seeds {out["SeedName"].n_unique()}  '
          f'targets {out["target"].n_unique()}  posts {out["n"].sum():,}', flush=True)
    print(f'date range: {out["bin"].min()} .. {out["bin"].max()}', flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.write_parquet(args.out, compression='zstd')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
