"""Forwards unadapted UNets on in-distribution and shifted val data; saves per-sample band powers for the start gate."""
import copy
import sys

import numpy as np
import torch

from msc.tta import setup
from msc.tta.eval import eval as ev

PAIRS = {"3z5bxjzp": 100, "qr6rs0jb": 500}
SHELLS = 33


def forward(run_id, op_re, data_re, device):
    """Returns per-sample pred/gt power and residual power at the operator's viscosity, shells < SHELLS.

    Args:
      run_id: pretraining wandb run id.
      op_re: the operator's training Reynolds number.
      data_re: Reynolds number of the val data scored.
      device: torch device.

    Returns:
      Dict of float32 arrays (N, SHELLS, T) / (N, SHELLS, T-2).
    """
    model, cfg = setup.load_model(run_id, device)
    cfg = copy.deepcopy(cfg)
    cfg.data.data_path = setup.data_path_for_re(data_re)
    out = ev.forward_bands(model, setup.build_dataset(cfg, "val"), device,
                           regime=setup.Regime(op_re, data_re), time_scale=cfg.data.time_scale,
                           temporal_pad=cfg.data.temporal_pad, pad_mode=cfg.data.pad_mode,
                           t_interval=cfg.loss.t_interval)
    return {k: out[k][:, :SHELLS].astype(np.float32) for k in ("pred_pt", "gt_pt", "pde_res_pred_op_pt")}


if __name__ == "__main__":
    device = torch.device("cuda")
    arrays = {}
    for run_id, op_re in PAIRS.items():
        for data_re in PAIRS.values():
            for k, v in forward(run_id, op_re, data_re, device).items():
                arrays[f"{run_id}_re{data_re}_{k}"] = v
            print(run_id, data_re, "done", flush=True)
    np.savez_compressed(sys.argv[1], **arrays)
