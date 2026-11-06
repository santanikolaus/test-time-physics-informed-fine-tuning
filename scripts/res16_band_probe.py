"""Band + time-resolved k<=7 probe for the 4 Re100 16² runs (FNO/UNet × spectral/strided).

Same machinery as scripts/time_band_resolved.py: load guard (tta_eval.probe ->
val_l2 reproduction) then tta_eval.band_eval on the held-out test split [260:300],
op_re=test_re=100. Adds the per-Chebyshev-shell early/late split from the joint
err_pt(band,T) / gt_pt(band,T) so the loss is dissected over BOTH axes like the
banked tables. val_l2 must reproduce training val (0.344/0.378/0.246/0.293) — that
is the correctness anchor (wrong forward settings break it).

    CUDA_VISIBLE_DEVICES=0 python scripts/res16_band_probe.py --device cuda
"""
import argparse
import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import eval as tta_eval
from msc.tta import setup as tta_setup

NS = "/system/user/studentwork/wehofer/data/ns"
SUB_T = 2
VAL_OFFSET, VAL_N = 200, 60
TEST_OFFSET, TEST_N = 260, 40
CK = tta_setup.ROOT / "pretrain-kol"

FNO_CFG = dict(tta_setup.MODEL_CFG)                       # op100 arch: n8 h128 lift0
UNET_CFG = dict(model_arch="unet", data_channels=4, out_channels=1,
                base_channels=64, depth=1, temporal_mixer="spatial",
                temporal_mixer_modes=8, spatial_mixer_hidden=64)

# (name, cfg, ckpt_id, data_file, train_val_l2)
RUNS = [
    ("FNO·spectral",  FNO_CFG,  "vsm4xtg5", "NS_fine_Re100_T128_res16_part0.npy",         0.344),
    ("FNO·strided",   FNO_CFG,  "seeh9q04", "NS_fine_Re100_T128_res16_strided_part0.npy", 0.378),
    ("UNet·spectral", UNET_CFG, "05wrabf5", "NS_fine_Re100_T128_res16_part0.npy",         0.246),
    ("UNet·strided",  UNET_CFG, "vfhirgsi", "NS_fine_Re100_T128_res16_strided_part0.npy", 0.293),
]


def load_model(cfg, ckpt, device):
    model = build_fno_kf(cfg)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def shell_split(err_pt, gt_pt, nE):
    """Per-shell early/late rel-L2 from joint band×time error/gt power."""
    early = np.sqrt(err_pt[:, 1:1 + nE].sum(1) / (gt_pt[:, 1:1 + nE].sum(1) + 1e-30))
    late = np.sqrt(err_pt[:, -nE:].sum(1) / (gt_pt[:, -nE:].sum(1) + 1e-30))
    return early, late


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="scripts/outputs/res16_band_probe.npz")
    args = p.parse_args()
    device = torch.device(args.device)

    rows, save = [], {}
    for name, cfg, ck_id, data_file, train_v in RUNS:
        ckpt = CK / ck_id / "checkpoints" / "best.ckpt"
        data = f"{NS}/{data_file}"
        ds_val = KFDataset(data, n_samples=VAL_N, offset=VAL_OFFSET, sub_t=SUB_T)
        ds_test = KFDataset(data, n_samples=TEST_N, offset=TEST_OFFSET, sub_t=SUB_T)
        S, T = ds_val[0]["y"].shape[0], ds_val[0]["y"].shape[-1]
        print(f"\n===== {name} =====  S={S} T={T}  ckpt={ck_id}", flush=True)

        model = load_model(cfg, str(ckpt), device)
        val_l2 = float(tta_eval.probe(model, ds_val, device, nu=100)["val_l2"].mean())
        guard = "OK" if abs(val_l2 - train_v) < 0.01 else f"MISMATCH(train {train_v})"
        print(f"  LOAD GUARD val_l2={val_l2:.4f}  [{guard}]", flush=True)

        res = tta_eval.band_eval(model, ds_test, device, op_re=100, test_re=100)
        e_k, l_k = shell_split(res["err_pt"], res["gt_pt"], res["nE"])
        print(f"  TEST[260:300] err_full={res['err_full']:.4f} err_k7={res['err_k7']:.4f}"
              f" | k<=7 early={res['early']:.4f} late={res['late']:.4f} ratio={res['ratio']:.2f}")
        print("  per-shell late k:" + " ".join(f"k{k}={l_k[k]:.3f}" for k in range(min(8, len(l_k)))))
        rows.append((name, val_l2, res["err_full"], res["err_k7"], res["early"], res["late"], res["ratio"]))
        save[f"{name}__err_pt"] = res["err_pt"]; save[f"{name}__gt_pt"] = res["gt_pt"]
        save[f"{name}__shell_early"] = e_k; save[f"{name}__shell_late"] = l_k

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **save)
    print("\n" + "=" * 86)
    print(f"{'run':<15}{'val_l2':>8}{'err_full':>10}{'err_k7':>9}{'early':>9}{'late':>8}{'late/early':>11}")
    print("-" * 86)
    for r in rows:
        print(f"{r[0]:<15}{r[1]:>8.4f}{r[2]:>10.4f}{r[3]:>9.4f}{r[4]:>9.4f}{r[5]:>8.4f}{r[6]:>11.2f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
