"""Seed a sweep with the prior runs it is able to express.

wandb refuses a prior whose value for a categorical parameter is not in that
sweep's list, so narrowing a range also narrows what history can be carried in.
"""
import sys
import yaml
import wandb

ENTITY, PROJECT = "benjamin-steel-projects", "potential_landscape_training"


def eligible(cfg, runs):
    cats = {k: v["values"] for k, v in cfg["parameters"].items() if "values" in v}
    keep, dropped = [], {}
    for r in runs:
        bad = [k for k, vals in cats.items()
               if k in r.config and r.config[k] not in vals]
        if bad:
            for k in bad:
                dropped[k] = dropped.get(k, 0) + 1
        else:
            keep.append(r)
    return keep, dropped


def main(path, source_sweeps, create):
    cfg = yaml.safe_load(open(path))
    api = wandb.Api()
    runs = []
    for sid in source_sweeps:
        runs += [r for r in api.sweep(f"{ENTITY}/{PROJECT}/{sid}").runs
                 if r.summary.get("objective") is not None]
    keep, dropped = eligible(cfg, runs)
    print(f"{path}: {len(keep)} of {len(runs)} prior runs are expressible")
    if dropped:
        print("  excluded by:", ", ".join(f"{k} ({n})" for k, n in
                                          sorted(dropped.items(), key=lambda x: -x[1])))
    best = sorted((r.summary["objective"] for r in keep), reverse=True)[:3]
    print("  best carried objectives:", ", ".join(f"{b:.5f}" for b in best) or "none")
    if create:
        sid = wandb.sweep(cfg, entity=ENTITY, project=PROJECT,
                          prior_runs=[r.id for r in keep])
        print("NEW_SWEEP", sid)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), "--create" in sys.argv)
