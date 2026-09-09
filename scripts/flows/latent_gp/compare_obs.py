"""Properties of the exported latent under each observation model.

The landscape model consumes these trajectories, and what it needs is motion it
can predict. So this reports, per observation model, how far the latent travels
over the scoring horizon relative to its own spread, how much of that travel is
the reverse of the previous step -- the momentum baseline the landscape model
has to beat -- and how uncertain the state is.

Both the smoothed and the filtered state are reported. Smoothing induces
negative autocorrelation in the increments by itself, which reads as mean
reversion, so a momentum figure only says something about the data if it
survives on the filtered state.

Needs no GPU, so it can be run while one is busy.

    python -m latent_gp.compare_obs tmp/gpfa_cells_sub10.parquet.zstd
"""

import argparse
import sys

import numpy as np
import polars as pl

from . import cells as gp_cells
from .latents import LatentConfig, build_latents, coord_cols


def _pairs_at_horizon(t, z, horizon):
    """Displacement over `horizon` days, and the reverse of the preceding one."""
    j = np.searchsorted(t, t + horizon)
    i = np.searchsorted(t, t - horizon)
    ok = (j < len(t)) & (i >= 0) & (np.arange(len(t)) > 0)
    if not ok.any():
        return None
    k = np.flatnonzero(ok)
    return z[j[k]] - z[k], -(z[k] - z[i[k]])


def props(df, n_dims, horizon, filtered=False):
    coord, causal, sd_col = coord_cols(n_dims)
    col = causal if filtered else coord
    x = np.stack(df[col].to_numpy())

    d_all, m_all = [], []
    for _, g in df.sort('createtime').group_by('filter_value', maintain_order=True):
        t = g['createtime'].to_numpy().astype('datetime64[s]').astype(float) / 86400.0
        if len(t) < 3:
            continue
        got = _pairs_at_horizon(t, np.stack(g[col].to_numpy()), horizon)
        if got is not None:
            d_all.append(got[0]); m_all.append(got[1])
    d = np.concatenate(d_all)
    m = np.concatenate(m_all)

    level_sd = float(x.std(0).mean())
    step_rms = float(np.sqrt((d ** 2).sum(1).mean()))
    return {
        'level_sd': level_sd,
        'post_sd': float(np.stack(df[sd_col].to_numpy()).mean()),
        'step_rms': step_rms,
        'step_over_level': step_rms / (level_sd * np.sqrt(n_dims)),
        'momentum_rho': float((d * m).sum()
                              / np.sqrt((d ** 2).sum() * (m ** 2).sum())),
        'pairs': len(d),
    }


def main():
    import splits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path')
    ap.add_argument('--obs-models', default='hard,soft,mixture')
    ap.add_argument('--n-dims', type=int, default=6)
    ap.add_argument('--n-fast', type=int, default=1)
    ap.add_argument('--fast-tau', type=float, default=80.0)
    ap.add_argument('--bin-factor', type=int, default=4)
    ap.add_argument('--interp-days', type=float, default=8.0)
    ap.add_argument('--holdout-days', type=int, default=182)
    ap.add_argument('--horizon-days', type=float, default=90.0)
    ap.add_argument('--calibration', default='')
    ap.add_argument('--temperature', type=float, default=1.0,
                    help='read by soft and mixture only')
    args = ap.parse_args()

    spec = splits.SplitSpec(holdout_days=args.holdout_days)
    seed_split = splits.seed_split(gp_cells.seed_names(args.path), spec)

    rows = []
    for obs in args.obs_models.split(','):
        df = build_latents(
            LatentConfig(cells_path=args.path, n_dims=args.n_dims, n_fast=args.n_fast,
                         fast_tau=args.fast_tau, bin_factor=args.bin_factor,
                         interp_days=args.interp_days, min_target_volume=400,
                         obs_model=obs, obs_temperature=args.temperature,
                         calibration_path=args.calibration),
            spec, seed_split, log=lambda *a: print('   ', *a, flush=True))
        for filtered in (False, True):
            r = props(df, args.n_dims, args.horizon_days, filtered)
            label = obs if args.temperature == 1.0 else f'{obs}@T{args.temperature:g}'
            rows.append({'obs': label,
                         'state': 'causal' if filtered else 'smoothed', **r})
            print(f'  {rows[-1]["obs"]:12} {rows[-1]["state"]:9} '
                  + '  '.join(f'{k}={v:.4g}' for k, v in r.items()), flush=True)

    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=4):
        print(pl.DataFrame(rows))


if __name__ == '__main__':
    sys.path.insert(0, 'scripts/flows')
    main()
