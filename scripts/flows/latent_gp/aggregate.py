"""Build the cell-level stance aggregate from the weekly classifier output.

A cell is one (seed, target, 2-day bin). Each cell carries

  n, s_sum, s2_sum     the hard-label moments, which fix the three ordinal
                       category counts exactly
  q0 .. q<L-1>         posts falling on each lattice point of the probability
                       simplex, for the soft-evidence likelihood (probs.py)

so one aggregate serves all three observation models and they can be compared
without rebuilding it.

Weeks are aggregated to their own part file, so a rebuild after more weeks
classify only touches the new ones -- worth having while the classifier is
still running.

Run on prometheus.
"""

import glob
import os
import re

import numpy as np
import polars as pl

from . import probs

WEEK_RE = re.compile(r'(\d{4})_(\d{1,2})_doc_targets')
STANCE_COLUMNS = ['id', 'platform', 'createtime', 'Targets', 'Stances', 'seed']
BIN = '2d'

# the actor types the trajectory model is about; media and state accounts are
# excluded among foreign seeds because they are not individuals holding stances
KEEP_TYPES = ['politician', 'influencer']
DROP_FOREIGN_SUBTYPES = ['media', 'state']


def week_of(path):
    m = WEEK_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def weeks_with_probs(stance_dir, probs_dir):
    """Weeks carrying both a stance file and a probabilities file.

    A week with labels but no probabilities is dropped rather than filled in
    from its labels: mixing the two would make the observation model a function
    of time, and the time split could no longer be read as one.
    """
    stance = {week_of(p): p for p in glob.glob(os.path.join(stance_dir, '*doc_targets_with_stance*'))}
    prob = {week_of(p): p for p in glob.glob(os.path.join(probs_dir, '*doc_targets_stance_probs*'))}
    both = sorted(k for k in stance if k in prob and k is not None)
    return [(k, stance[k], prob[k]) for k in both], sorted(set(stance) - set(prob) - {None})


def _pairs(stance_path, probs_path):
    """One row per (post, target) with a seed, a hard stance and a probability vector."""
    df = pl.read_parquet(stance_path, columns=STANCE_COLUMNS)
    df = df.unique(['id', 'platform']).with_columns([
        pl.col('seed').struct.field('SeedName').alias('SeedName'),
        pl.col('seed').struct.field('MainType').alias('MainType'),
        pl.col('seed').struct.field('SubType').alias('SubType'),
    ]).drop('seed')
    df = df.filter(
        pl.col('MainType').is_in(KEEP_TYPES)
        | ((pl.col('MainType') == 'foreign')
           & (~pl.col('SubType').is_in(DROP_FOREIGN_SUBTYPES)))
    ).filter(pl.col('SeedName') != '')
    if not len(df):
        return None
    df = df.explode(['Targets', 'Stances']).drop_nulls(['Targets', 'Stances'])
    if not len(df):
        return None

    pr = pl.read_parquet(probs_path).explode(['Targets', 'Probs']) \
        .drop_nulls(['Targets', 'Probs']).unique(['id', 'platform', 'Targets'])
    df = df.join(pr, on=['id', 'platform', 'Targets'], how='inner')
    return df if len(df) else None


def _aggregate_week(df, resolution):
    """Per-cell hard moments and lattice counts for one week's pairs."""
    df = df.with_columns([
        pl.col('Stances').replace_strict(probs.STANCE_VALUE, default=None).alias('s'),
        pl.col('createtime').dt.replace_time_zone(None).dt.truncate(BIN).alias('bin'),
        pl.col('Probs').list.to_array(3).alias('p'),
    ]).drop_nulls('s')
    if not len(df):
        return None

    idx = probs.assign(df['p'].to_numpy(), resolution)
    df = df.with_columns(pl.Series('arch', idx))
    L = probs.n_archetypes(resolution)
    return df.rename({'Targets': 'target'}).group_by(['SeedName', 'target', 'bin']).agg(
        [pl.col('s').sum().alias('s_sum'),
         (pl.col('s') ** 2).sum().alias('s2_sum'),
         pl.len().alias('n')]
        + [(pl.col('arch') == i).sum().cast(pl.Float64).alias(f'q{i}') for i in range(L)])


def build_parts(stance_dir, probs_dir, parts_dir, resolution, log=print):
    """Aggregate every week that has probabilities and is not already done."""
    ready, missing = weeks_with_probs(stance_dir, probs_dir)
    if missing:
        log(f'{len(missing)} weeks have labels but no probabilities, skipped: '
            f'{missing[0]} .. {missing[-1]}')
    os.makedirs(parts_dir, exist_ok=True)
    log(f'{len(ready)} weeks with probabilities')

    for i, ((year, week), s_path, p_path) in enumerate(ready):
        out = os.path.join(parts_dir, f'{year}_{week:02d}_cells.parquet.zstd')
        if os.path.exists(out):
            continue
        pairs = _pairs(s_path, p_path)
        agg = None if pairs is None else _aggregate_week(pairs, resolution)
        if agg is None:
            log(f'  {year}_{week}: no usable pairs')
            continue
        agg.write_parquet(out, compression='zstd')
        if (i + 1) % 10 == 0:
            log(f'  {i + 1}/{len(ready)} weeks')
    return len(ready)


def merge_parts(parts_dir, cache, log=print):
    """Sum the part files into one aggregate.

    A 2-day bin can straddle two weekly files, so the parts have to be summed
    rather than concatenated. Streamed, because the concatenation does not fit
    in memory at full scale.
    """
    parts = sorted(glob.glob(os.path.join(parts_dir, '*_cells.parquet.zstd')))
    if not parts:
        raise ValueError(f'no part files in {parts_dir}')
    lf = pl.scan_parquet(parts)
    value_cols = [c for c in lf.collect_schema().names()
                  if c not in ('SeedName', 'target', 'bin')]
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    (lf.group_by(['SeedName', 'target', 'bin'])
       .agg([pl.col(c).sum() for c in value_cols])
       .sink_parquet(cache, compression='zstd'))
    log(f'wrote {cache} from {len(parts)} parts')


def build(stance_dir, probs_dir, parts_dir, cache, resolution=6, log=print):
    if os.path.exists(cache):
        log(f'using cached aggregate {cache}')
        return
    build_parts(stance_dir, probs_dir, parts_dir, resolution, log=log)
    merge_parts(parts_dir, cache, log=log)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stance-dir', default='data/stance_targets/2022-01-01-onwards_noun_phrase_stance')
    ap.add_argument('--probs-dir', default=None, help='defaults to <stance-dir>_probs')
    ap.add_argument('--parts-dir', default='tmp/gpfa_cell_parts')
    ap.add_argument('--out', default='tmp/gpfa_cells_2022_onwards.parquet.zstd')
    ap.add_argument('--resolution', type=int, default=6)
    args = ap.parse_args()
    probs_dir = args.probs_dir or f'{args.stance_dir}_probs'

    build(args.stance_dir, probs_dir, args.parts_dir, args.out, args.resolution,
          log=lambda *a: print(*a, flush=True))

    lf = pl.scan_parquet(args.out)
    print(lf.select(pl.len().alias('cells'), pl.col('n').sum().alias('posts'),
                    pl.col('SeedName').n_unique().alias('seeds'),
                    pl.col('target').n_unique().alias('targets'),
                    pl.col('bin').min().alias('lo'),
                    pl.col('bin').max().alias('hi')).collect(), flush=True)


if __name__ == '__main__':
    main()
