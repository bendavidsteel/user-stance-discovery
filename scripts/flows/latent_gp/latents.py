"""Split-aware latent trajectories for the landscape model.

Replaces impute + PPCA + rolling mean: the latent is smooth in time by
construction, so no post-hoc smoothing window is needed.

The split applies at two levels, and conflating them is what leaks:

  W, b, c   the global representation -- fit on training trajectories inside
            the training period only, because these are shared across seeds
            and so can carry held-out information into every prediction. W is
            the per-target loading matrix, exported by `build_loadings`.

  z_m(t)    per-seed state -- inferred for every trajectory over the whole
            period with the global parameters frozen. This is measurement, not
            prediction: a held-out trajectory is allowed to be *observed*, it
            just must not shape the representation.

The smoothed state at t depends on observations after t, which inflates skill
at horizons short relative to the latent's own timescale. `causal_*` columns
hold the filtered state instead, which has no such dependence.

A rolled-back origin (`spec.origin_offset_days`) also truncates the grid at the
window's end, so neither state sees past the period being scored.

`obs_model` chooses how much of the stance classifier's output the fit sees:
its label ('hard'), its label through a measured error channel ('channel'), its
probabilities as expected counts ('soft'), or its probabilities as soft
evidence ('mixture'). See probs.py, and calibrate.py for the channel.
"""

import dataclasses
import datetime
import hashlib
import json
import os

import numpy as np
import polars as pl

from . import aggregate, calibrate, cells, fit as fit_mod, probs


# The singularity guard m_step has always used. At this value the loadings are
# unregularised -- per-target site precision runs from ~1e2 up -- so it is what
# every fit before the prior existed used, and the tag omits it to keep those
# fits addressable.
W_RIDGE_OFF = 1e-4

@dataclasses.dataclass(frozen=True)
class LatentConfig:
    cells_path: str
    n_dims: int = 6
    n_fast: int = 2
    fast_tau: float = 80.0
    fast_kind: str = 'wiener'       # wiener (diffusing) or ou (confined)
    slow_kind: str = 'const'
    slow_tau: float = 2560.0
    bin_factor: int = 8
    rho: float = 0.0
    iters: int = 25
    infer_iters: int = 15
    min_target_volume: int = 400
    interp_days: float = 0.0        # 0 = keep the native bin grid
    obs_model: str = 'hard'         # hard | channel | soft | mixture
    obs_temperature: float = 1.0    # >1 flattens the classifier posterior
    prob_resolution: int = 6        # simplex lattice spacing is 1/resolution
    prob_floor: float = 0.01        # uniform mass mixed into each lattice point
    calibration_path: str = ''      # calibrate.py output; required by 'channel'
    w_ridge: float = W_RIDGE_OFF    # prior precision on the loadings
    seed: int = 0

    def __post_init__(self):
        # A sweep passes fast_tau=40 where the default is 80.0, and hydra keeps
        # it an int. The tag hashes the value's repr, so without coercion the
        # same configuration keys two different cache entries and every trial
        # refits from scratch.
        for f in dataclasses.fields(self):
            cast = {int: int, float: float, 'int': int, 'float': float}.get(f.type)
            if cast is not None:
                object.__setattr__(self, f.name, cast(getattr(self, f.name)))

    @classmethod
    def from_cfg(cls, cfg):
        """Read the hydra config, where n_dims and min_target_volume are top level."""
        return cls(
            cells_path=cfg.latents.cells_path,
            n_dims=cfg.n_dims,
            n_fast=cfg.latents.n_fast,
            fast_tau=cfg.latents.fast_tau,
            fast_kind=cfg.latents.fast_kind,
            slow_kind=cfg.latents.slow_kind,
            slow_tau=cfg.latents.slow_tau,
            bin_factor=cfg.latents.bin_factor,
            interp_days=cfg.latents.interp_days,
            rho=cfg.latents.rho,
            iters=cfg.latents.iters,
            infer_iters=cfg.latents.infer_iters,
            min_target_volume=cfg.min_target_volume,
            obs_model=cfg.latents.obs_model,
            obs_temperature=cfg.latents.obs_temperature,
            prob_resolution=cfg.latents.prob_resolution,
            prob_floor=cfg.latents.prob_floor,
            calibration_path=cfg.latents.calibration_path,
            w_ridge=cfg.latents.get('w_ridge', W_RIDGE_OFF),
            seed=cfg.latents.seed,
        )

    @property
    def tag(self):
        """Cache key: everything that changes the fitted latents.

        Fields the chosen observation model never reads are left out, so a sweep
        varying a probability setting still hits the cache on the trials that
        ignore it, instead of refitting the same latents under a new key.
        """
        skip = ({'cells_path'} | _unused_by(self.obs_model)
                | _unused_by_mix(self.n_fast, self.n_dims, self.slow_kind))
        if self.w_ridge == W_RIDGE_OFF:
            skip = skip | {'w_ridge'}   # every fit that predates the prior
        body = '|'.join(f'{f.name}={getattr(self, f.name)}'
                        for f in dataclasses.fields(self) if f.name not in skip)
        return hashlib.blake2b(body.encode(), digest_size=6).hexdigest()


# the two frames one fit produces, as indices into what _fit returns
LATENTS, LOADINGS = 0, 1

OBS_MODELS = ('hard', 'channel', 'soft', 'mixture')
_PROB_FIELDS = frozenset({'obs_temperature', 'prob_resolution', 'prob_floor'})
_CHANNEL_FIELDS = frozenset({'calibration_path'})


_FAST_FIELDS = frozenset({'fast_kind', 'fast_tau'})
_SLOW_FIELDS = frozenset({'slow_kind', 'slow_tau'})


def _unused_by(obs_model):
    if obs_model == 'hard':
        return _PROB_FIELDS | _CHANNEL_FIELDS
    if obs_model == 'channel':
        return _PROB_FIELDS
    return _CHANNEL_FIELDS


def _unused_by_mix(n_fast, n_dims, slow_kind):
    """Timescale settings the chosen mix never reads.

    Without this every value of an unread setting keys its own cache entry, and
    at full scale a latent fit is 12-20 minutes.
    """
    unused = set()
    if n_fast >= n_dims:
        unused |= _SLOW_FIELDS
    elif n_fast <= 0:
        unused |= _FAST_FIELDS
    if slow_kind == 'const':
        unused.add('slow_tau')          # a frozen dimension has no timescale
    return frozenset(unused)


def observation(df, train_mask, obs_model, temperature=1.0, resolution=6,
                floor=0.01, calibration_path=''):
    """Category counts per the observation model, plus its lattice and log ratios.

    'soft' and 'mixture' read the same lattice counts, so they differ in the
    likelihood alone rather than in how much of the classifier's output
    survived quantisation. pi is a property of the classifier and global to the
    fit, so like W and b it is taken from training cells only.
    """
    if obs_model not in OBS_MODELS:
        raise ValueError(f'obs_model must be one of {OBS_MODELS}')
    if obs_model == 'hard':
        return df, None, None

    if obs_model == 'channel':
        if not calibration_path:
            raise ValueError("obs_model 'channel' needs calibration_path: run "
                             'latent_gp.calibrate on a coded sample first')
        with open(calibration_path) as fh:
            cal = json.load(fh)
        n_arch = calibrate.corner_counts(df['n_neg'].to_numpy(),
                                         df['n_neu'].to_numpy(),
                                         df['n_pos'].to_numpy())
        return df, n_arch, calibrate.channel_log_ratio(cal['confusion'])

    q = probs.archetypes(resolution, temperature, floor)
    counts = cells.lattice(df, resolution)
    soft = counts @ q
    df = df.with_columns(pl.Series('n_neg', soft[:, 0]),
                         pl.Series('n_neu', soft[:, 1]),
                         pl.Series('n_pos', soft[:, 2]))
    if obs_model == 'soft':
        return df, None, None
    tr = soft[train_mask]
    pi = probs.marginal(tr[:, 0], tr[:, 1], tr[:, 2])
    return df, counts, probs.log_likelihood_ratio(q, pi)


def data_tag(path):
    """Fingerprint of the cell aggregate.

    The config tag deliberately ignores cells_path, so that moving the file does
    not invalidate the cache -- but then rebuilding the aggregate in place has
    to. Reads the provenance aggregate.py records beside the file, and hashes
    the file itself when there is none. Either way it tracks the contents:
    a row count and a column count do not move when a week is reclassified.
    """
    body = aggregate.read_sidecar(path + aggregate.SIDECAR) \
        or aggregate.file_digest(path)
    return hashlib.blake2b(body.encode(), digest_size=4).hexdigest()


def coord_cols(n_dims):
    return f'coord_{n_dims}d', f'causal_{n_dims}d', f'sd_{n_dims}d'


def _standardise(Ez, train_cells):
    """Centre and scale each dimension using training-region state only.

    W is identified only up to scale, so without this the latent's units drift
    between sweep configurations and any absolute-scale hyperparameter (sigma,
    the confinement threshold) has to be re-tuned for each one.
    """
    ref = Ez[train_cells[:, 0], train_cells[:, 1]]
    mu = ref.mean(0)
    sd = np.maximum(ref.std(0), 1e-6)
    return mu, sd


def _loading_frame(meta, W, b, mu, sd):
    """Per-target loadings in the units the exported latent is reported in.

    The frame carries (z - mu) / sd, so the loadings that reconstruct a cell's
    f from those coordinates are W * sd with mu folded into the intercept. In
    the fit's own units they are off by a per-dimension factor -- invisible
    within a dimension, wrong across them.
    """
    return pl.DataFrame({
        'target': meta['targets'],
        'loading': W * sd,
        'intercept': b + W @ mu,
    })


def _cache_paths(lcfg, spec, cache_dir):
    """Where this configuration's latents and loadings live, or (None, None)."""
    if not cache_dir:
        return None, None
    key = f'{lcfg.tag}_{data_tag(lcfg.cells_path)}_{spec.tag}'
    return (os.path.join(cache_dir, f'latents_{key}.parquet.zstd'),
            os.path.join(cache_dir, f'loadings_{key}.parquet.zstd'))


def build_latents(lcfg, spec, seed_split, cache_dir=None, log=print):
    """Fit the latent-GP factor model under `spec` and return per-bin states.

    `seed_split` maps trajectory id -> 'train' / 'val' / 'test'. Returns a frame
    of (createtime, filter_value, coord, causal coord, posterior sd, n_posts).
    """
    return _build(lcfg, spec, seed_split, cache_dir, log, LATENTS)


def build_loadings(lcfg, spec, seed_split, cache_dir=None, log=print):
    """Per-target loadings from the same fit: (target, loading, intercept).

    One row per target surviving min_target_volume, in the order W was fitted
    in, so `loading_matrix` can hand the pair to code written against PCA
    components. Reported in the units build_latents reports, not the fit's.
    """
    return _build(lcfg, spec, seed_split, cache_dir, log, LOADINGS)


def _build(lcfg, spec, seed_split, cache_dir, log, want):
    """Whichever output was asked for, refitting only when that one is absent.

    Latents cached before the loadings existed are still valid on their own, so
    asking for them never refits for the sake of the sibling file.
    """
    paths = _cache_paths(lcfg, spec, cache_dir)
    if paths[want] and os.path.exists(paths[want]):
        log(f'reusing cached {paths[want]}')
        return pl.read_parquet(paths[want])

    built = _fit(lcfg, spec, seed_split, log)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for path, frame in zip(paths, built):
            frame.write_parquet(path, compression='zstd')
    return built[want]


def loading_matrix(loadings):
    """(K, J) loadings and the target names indexing their columns.

    Transposed from the fit's (J, K) W: dimension-first is the orientation PCA
    components come in, and what the analysis scripts index by.
    """
    return loadings['loading'].to_numpy().T, loadings['target'].to_list()


def _fit(lcfg, spec, seed_split, log):
    """Returns (per-bin states, per-target loadings)."""
    df, meta = cells.load(lcfg.cells_path, lcfg.bin_factor,
                          min_target_volume=lcfg.min_target_volume)
    K = lcfg.n_dims
    # both bin indices come from the untruncated grid, then the grid is cut to
    # the window's end so no state is informed by anything after it
    t_cut, cutoff = cells.cutoff_bin(meta, spec.holdout_days, spec.origin_offset_days)
    t_end = cells.window_end_bin(meta, spec.origin_offset_days)
    df, meta = cells.truncate(df, meta, t_end)
    is_train = np.array([seed_split.get(s, 'test') == 'train' for s in meta['seeds']])
    log(f"M={meta['M']} J={meta['J']} T={meta['T']} K={K} "
        f"train seeds {int(is_train.sum())} cutoff {cutoff:%Y-%m-%d} (bin {t_cut})")

    comps = fit_mod.prior_components(K, lcfg.n_fast, lcfg.fast_tau,
                                     lcfg.slow_kind, lcfg.slow_tau,
                                     fast_kind=lcfg.fast_kind)

    m_arr = df['m'].to_numpy()
    t_arr = df['t'].to_numpy()
    train_mask = is_train[m_arr] & (t_arr < t_cut)
    if not train_mask.any():
        raise ValueError('no training cells: check holdout_days and '
                         'origin_offset_days against the data span')

    df, n_arch, logL = observation(
        df, train_mask, lcfg.obs_model, lcfg.obs_temperature,
        lcfg.prob_resolution, lcfg.prob_floor, lcfg.calibration_path)
    tr_arch = None if n_arch is None else n_arch[train_mask]
    d_train = cells.pack(cells.deflate(df.filter(pl.Series(train_mask)), lcfg.rho),
                         meta, tr_arch)
    log(f'fitting on {len(d_train["j"]):,} training cells, obs {lcfg.obs_model}')
    r = fit_mod.fit(d_train, comps, meta['dt'], K, lcfg.iters, seed=lcfg.seed,
                    logL=logL, w_ridge=lcfg.w_ridge, log=log)

    d_all = cells.pack(cells.deflate(df, lcfg.rho), meta, n_arch)
    Ez, Ezz = fit_mod.infer(d_all, comps, meta['dt'], K,
                            r['W'], r['b'], r['c'], lcfg.infer_iters, logL=logL)
    Ez_c, _ = fit_mod.infer(d_all, comps, meta['dt'], K,
                            r['W'], r['b'], r['c'], lcfg.infer_iters,
                            filtered=True, logL=logL)

    mu, sd = _standardise(Ez, np.stack([m_arr[train_mask], t_arr[train_mask]], 1))
    Ez = (Ez - mu) / sd
    Ez_c = (Ez_c - mu) / sd
    post_sd = np.sqrt(np.maximum(np.diagonal(Ezz, axis1=2, axis2=3), 0.0)) / sd

    out = _to_frame(df, meta, Ez, Ez_c, post_sd, K, lcfg.interp_days)
    out = out.join(pl.DataFrame({'filter_value': list(seed_split),
                                 'traj_split': list(seed_split.values())}),
                   on='filter_value', how='left')
    return out, _loading_frame(meta, r['W'], r['b'], mu, sd)


def _fine_grid(lo, hi, step):
    """Fractional bin positions per seed at `step` bins apart, plus their seed index."""
    n = np.floor((hi - lo) / step).astype(int) + 1
    offs = np.repeat(np.cumsum(np.r_[0, n[:-1]]), n)
    u = lo.repeat(n) + step * (np.arange(n.sum()) - offs)
    return np.repeat(np.arange(len(lo)), n), np.minimum(u, hi.repeat(n))


def _interpolate(Ez, m_i, u, hi_i):
    """Linear interpolation between bin centres.

    Exact for this prior family: the smoothed mean of a Wiener process between
    two grid points is the Brownian-bridge mean, which is linear in time, and a
    constant component is trivially linear. Nothing is being approximated.
    """
    t_lo = np.floor(u).astype(int)
    t_hi = np.minimum(t_lo + 1, hi_i)
    w = (u - t_lo)[:, None]
    return (1 - w) * Ez[m_i, t_lo] + w * Ez[m_i, t_hi]


def _to_frame(df, meta, Ez, Ez_c, post_sd, K, interp_days=0.0):
    """Rows spanning each seed's observed bins, optionally on a finer grid.

    Bins outside the span carry no data at all, so their state is pure prior
    and would be a fabricated observation for the landscape model.
    """
    span = df.group_by('m').agg(
        pl.col('t').min().alias('lo'), pl.col('t').max().alias('hi')).sort('m')
    lo = span['lo'].to_numpy().astype(float)
    hi = span['hi'].to_numpy().astype(float)
    seed_of = span['m'].to_numpy()

    step = (interp_days / meta['dt']) if interp_days > 0 else 1.0
    row, u = _fine_grid(lo, hi, step)
    m_i = seed_of[row]
    hi_i = hi[row].astype(int)
    t_i = np.floor(u).astype(int)

    posts = df.group_by(['m', 't']).agg(pl.col('n').sum().alias('n_posts'))
    origin = datetime.datetime(1970, 1, 1)
    offset = (meta['t0'] - origin).total_seconds() + 86400 * meta['dt'] / 2
    stamp = np.rint(1e6 * (offset + 86400 * meta['dt'] * u)).astype(np.int64)

    c_coord, c_causal, c_sd = coord_cols(K)
    out = pl.DataFrame({
        'm': m_i,
        't': t_i,
        'createtime': pl.Series(stamp).cast(pl.Datetime('us')),
        'filter_value': [meta['seeds'][m] for m in m_i],
        c_coord: _interpolate(Ez, m_i, u, hi_i),
        c_causal: _interpolate(Ez_c, m_i, u, hi_i),
        c_sd: post_sd[m_i, t_i],
    })
    return out.join(posts, on=['m', 't'], how='left') \
        .with_columns(pl.col('n_posts').fill_null(0.0)) \
        .drop(['m', 't']) \
        .sort(['filter_value', 'createtime'])
