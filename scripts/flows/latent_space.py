"""The representation the analysis scripts read: trajectories plus loadings.

`cfg.latents.method` picks between them. 'gpfa' fits the latent-GP factor
model here, inside the split boundary, exactly as the landscape model does --
the loadings come out of that same fit, so they describe the axes the
trajectories actually move along. The precomputed methods read coords and a
component matrix written over the whole dataset.

Loadings are returned (K, J) with the target names indexing the columns, which
is how PCA components arrive and so what code written against them expects.
"""

import datetime
import logging
import os

import numpy as np
import polars as pl

import splits
from latent_gp import (LatentConfig, build_latents, build_loadings, coord_cols,
                       loading_matrix)
from latent_gp import cells as gp_cells

logger = logging.getLogger(__name__)

# One name whatever the dimensionality, so nothing downstream hard-codes it
COORD = 'coord'

# The precomputed coords span the whole history; the analysis starts here. A
# gpfa aggregate is built over its own period, so applying this to it again
# would be a second, hidden opinion about the study window.
PRECOMPUTED_START = datetime.datetime(2022, 1, 1)


def load(cfg, smooth=True):
    """Returns (trajectories, (K, J) loadings, target names).

    The gpfa latent is smooth in time by construction, so `smooth` only ever
    applies to the precomputed coords -- averaging the latent again would just
    widen the window each point already covers.
    """
    if cfg.latents.method == 'gpfa':
        return _gpfa(cfg)
    return _precomputed(cfg, smooth)


def name(cfg):
    """Filename prefix for artifacts describing the chosen representation.

    Keeps a gpfa fit's dimension labels off the precomputed method's, whose
    axes are a different basis over a different target set.
    """
    return 'gpfa' if cfg.latents.method == 'gpfa' else cfg.dim_reduction_method


def n_moving_dims(cfg):
    """Dimensions that actually drift, which is all a mover ranking can use.

    Under the gpfa prior only the first n_fast dimensions have drifting
    dynamics; the rest are frozen per trajectory, so their movement is zero by
    construction rather than by measurement.
    """
    if cfg.latents.method != 'gpfa':
        return cfg.n_dims
    return min(cfg.latents.n_fast, cfg.n_dims)


def _gpfa(cfg):
    spec = splits.SplitSpec.from_cfg(cfg)
    lcfg = LatentConfig.from_cfg(cfg)
    seed_split = splits.seed_split(gp_cells.seed_names(lcfg.cells_path), spec)
    kw = dict(cache_dir=cfg.latents.cache_dir, log=logger.info)

    coord, causal, _ = coord_cols(lcfg.n_dims)
    state = causal if cfg.latents.causal_state else coord
    target_df = build_latents(lcfg, spec, seed_split, **kw) \
        .select(['createtime', 'filter_value', pl.col(state).alias(COORD)]) \
        .sort(['filter_value', 'createtime'])

    components, targets = loading_matrix(build_loadings(lcfg, spec, seed_split, **kw))
    return target_df, components, targets


def _precomputed(cfg, smooth):
    trend_path = cfg.trend_path
    target_df = pl.read_parquet(
        os.path.join(trend_path, f'{cfg.dim_reduction_method}_coords.parquet.zstd'))
    coord_col = [c for c in target_df.columns if c.startswith('coord_')][0]
    n_dims = target_df.schema[coord_col].shape[0]

    target_df = target_df \
        .filter(pl.col('createtime') >= PRECOMPUTED_START) \
        .select(['createtime', 'filter_value', pl.col(coord_col).alias(COORD)]) \
        .sort(['filter_value', 'createtime'])
    if smooth:
        target_df = target_df \
            .with_columns([pl.col(COORD).arr.get(i).alias(f'dim_{i}')
                           for i in range(n_dims)]) \
            .rolling('createtime', period=f'{cfg.rolling_mean_window}d',
                     group_by='filter_value') \
            .agg([pl.col(f'dim_{i}').mean() for i in range(n_dims)]) \
            .with_columns(pl.concat_arr([f'dim_{i}' for i in range(n_dims)]).alias(COORD)) \
            .drop_nulls(COORD) \
            .select(['createtime', 'filter_value', COORD])

    component_df = pl.read_parquet(
        os.path.join(trend_path, f'{cfg.dim_reduction_method}_metadata.parquet.zstd'))
    if cfg.dim_reduction_method == 'sfa':
        components = component_df.filter(pl.col('n_components') == n_dims)['W'][0].to_numpy()
    elif cfg.dim_reduction_method in ['pca', 'ppca', 'pica']:
        components = np.stack(
            component_df.filter(pl.col('n_dims') == n_dims)['components'][0].to_numpy())
    else:
        raise ValueError(f'Unknown dim_reduction_method: {cfg.dim_reduction_method}')

    # these components are over the columns of the frame they were fitted on
    head = pl.read_parquet(
        os.path.join(trend_path, 'pivoted_and_imputed.parquet.zstd'), n_rows=1)
    targets = [c for c in head.columns if c not in ['createtime', 'filter_value']]
    assert len(targets) == components.shape[1]
    return target_df, components, targets
