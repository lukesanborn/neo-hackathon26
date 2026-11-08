"""One compact status line per tier for the knob sweep (used by a Monitor poll loop).

Reports every tier that has jobs recorded, so a screen round and a full round running at the
same time are both visible. Reads the Volume and the call-id file, so it is safe to run
repeatedly from anywhere and survives losing the launching terminal.

    python diag/watch_knobs.py            # one snapshot of every active tier
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag.launch_model_knobs as L  # noqa: E402


def tier_status(vol, modal, ids, tier: str, dataset: str) -> list[str]:
    """Counts + per-arm notes for one tier. Returns [] when that tier has no jobs recorded."""
    L.TIER, L.DATASET = tier, dataset              # Arm.tag / ft_key read these globals
    arms = L.ARMS
    if not any(k.endswith(a.tag) or f"|{a.tag}|" in k for k in ids for a in arms):
        return []

    def failed(key):
        """Only a crashed call counts; running/queued and unsubmitted both return None."""
        cid = ids.get(key)
        if not cid:
            return None
        try:
            modal.FunctionCall.from_id(cid).get(timeout=0)
        except TimeoutError:
            return None
        except Exception as e:
            return repr(e)[:110]
        return None

    # Completion counts come from the Volume, not the call-id file: concurrent chain processes can
    # drop each other's keys, and a missing key must never read as a missing result.
    seeds = [s for s in (1, 2)]
    pt_done = pt_fail = ft_sub = ft_done = ft_fail = 0
    notes = []
    for a in arms:
        if L.read_json(vol, f"{L.RUNS_ROOT}/{a.run_dir}/summary.json"):
            pt_done += 1
        elif (err := failed(f"pretrain|{a.tag}")):
            pt_fail += 1
            notes.append(f"    FAILED pretrain {tier}/{a.label}: {err}")
        for seed in seeds:
            key = L.ft_key(a, seed)
            m = L.read_json(vol, L.ft_metrics_path(a, seed))
            if m:
                ft_sub += 1
                ft_done += 1
                notes.append(f"    DONE {tier}/{a.label} s{seed}: test ΔRMSE {m['test']['delta_rmse']:.4f} "
                             f"| val {m.get('val_delta_rmse_selected', float('nan')):.4f} "
                             f"| {m.get('status')}")
            elif key in ids:
                ft_sub += 1
                if (err := failed(key)):
                    ft_fail += 1
                    notes.append(f"    FAILED finetune {tier}/{a.label} s{seed}: {err}")
    n = len(arms)
    head = (f"[{tier:6s}/{dataset:8s}] pretrain {pt_done}/{n} | finetune {ft_done}/{ft_sub} done of {n}")
    if pt_fail or ft_fail:
        head += f" | {pt_fail + ft_fail} FAILED"
    if pt_done == n and ft_done == n:
        head += "  <- COMPLETE"
    return [head] + notes[-6:]


def main():
    import modal

    vol = modal.Volume.from_name(L.VOL)
    ids = L.load_ids()
    out = []
    for tier in L.TIERS:
        for dataset in L.DATASETS:
            out += tier_status(vol, modal, ids, tier, dataset)
    print("\n".join(out) if out else "no jobs recorded yet")


if __name__ == "__main__":
    main()
