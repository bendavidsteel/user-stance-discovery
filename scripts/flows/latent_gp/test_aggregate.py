"""Rebuild decisions for the cell aggregate.

The question these ask is the one that went wrong: given sources that have
changed, does a rebuild actually rebuild? The per-week aggregation is stubbed
out, so what is under test is the staleness logic and not the schema.

Run as: python -m latent_gp.test_aggregate
"""

import os
import tempfile

import polars as pl

from . import aggregate as agg
from .latents import data_tag

WEEKS = [(2024, 1), (2024, 2), (2024, 3)]


def _sources(td):
    """A stance file and a probabilities file per week, with known contents."""
    s_dir = os.path.join(td, 'stance')
    p_dir = os.path.join(td, 'stance_probs')
    for d in (s_dir, p_dir):
        os.makedirs(d, exist_ok=True)
    for year, week in WEEKS:
        with open(os.path.join(s_dir, f'{year}_{week}_doc_targets_with_stance.parquet.zstd'), 'w') as fh:
            fh.write(f'stance {year} {week}')
        with open(os.path.join(p_dir, f'{year}_{week}_doc_targets_stance_probs.parquet.zstd'), 'w') as fh:
            fh.write(f'probs {year} {week}')
    return s_dir, p_dir


def _stub(monkey_built):
    """Stand in for the real per-week aggregation, recording what got built."""
    def _pairs(s_path, p_path):
        return open(s_path).read()

    def _aggregate_week(pairs, resolution):
        monkey_built.append(pairs)
        return pl.DataFrame({'SeedName': ['s'], 'target': ['t'],
                             'bin': [1], 'n': [float(len(pairs))]})
    return _pairs, _aggregate_week


def run(td, s_dir, p_dir, built):
    parts = os.path.join(td, 'parts')
    cache = os.path.join(td, 'cells.parquet.zstd')
    built.clear()
    agg.build(s_dir, p_dir, parts, cache, resolution=6, log=lambda *a: None)
    return parts, cache


def main():
    built = []
    real = (agg._pairs, agg._aggregate_week)
    agg._pairs, agg._aggregate_week = _stub(built)
    try:
        with tempfile.TemporaryDirectory() as td:
            s_dir, p_dir = _sources(td)

            parts, cache = run(td, s_dir, p_dir, built)
            assert len(built) == len(WEEKS), built
            assert os.path.exists(cache)
            first_tag = data_tag(cache)
            print(f'first build: {len(built)} weeks, data_tag {first_tag}')

            run(td, s_dir, p_dir, built)
            assert built == [], f'rebuilt {len(built)} unchanged weeks'
            assert data_tag(cache) == first_tag
            print('unchanged sources: nothing rebuilt, tag stable')

            # a touch must not be enough; the contents decide
            target = os.path.join(
                s_dir, '2024_2_doc_targets_with_stance.parquet.zstd')
            os.utime(target, (0, 0))
            run(td, s_dir, p_dir, built)
            assert built == [], 'a changed mtime forced a rebuild'
            print('touched source: still nothing rebuilt')

            with open(target, 'w') as fh:
                fh.write('stance 2024 2 reclassified')
            run(td, s_dir, p_dir, built)
            assert len(built) == 1, built
            assert 'reclassified' in built[0], built
            second_tag = data_tag(cache)
            assert second_tag != first_tag, 'the aggregate tag did not move'
            print(f'edited one week: 1 rebuilt, data_tag {first_tag} -> {second_tag}')

            # a part with no provenance cannot be claimed current
            os.remove(os.path.join(parts, '2024_03_cells.parquet.zstd') + agg.SIDECAR)
            run(td, s_dir, p_dir, built)
            assert len(built) == 1, built
            print('part with no recorded digest: rebuilt')

            # and a missing part is rebuilt even though its digest still matches
            os.remove(os.path.join(parts, '2024_01_cells.parquet.zstd'))
            run(td, s_dir, p_dir, built)
            assert len(built) == 1, built
            print('deleted part: rebuilt')
    finally:
        agg._pairs, agg._aggregate_week = real

    print('\nall aggregate rebuild checks passed')


def test_aggregate_rebuilds():
    main()


if __name__ == '__main__':
    main()
