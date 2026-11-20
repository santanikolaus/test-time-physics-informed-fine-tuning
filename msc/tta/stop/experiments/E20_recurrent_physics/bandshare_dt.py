"""E20 R10: GT residual power per band at dt 1/64 vs 1/128 (sub_t 2/1), stencil o2/o4, res 128/256, val chains, Re BANDSHARE_RE (CPU)."""
import os

import numpy as np
import torch

from msc.tta.eval.eval import band_power, cheb_bins
from src.pde.ns import NSVorticity

RE = int(os.environ.get("BANDSHARE_RE", 500))
RESS = [int(r) for r in os.environ.get("BANDSHARE_RES", "128,256").split(",")]
D = f"/system/user/studentwork/wehofer/data/ns/NS_fine_Re{RE}_T128_res{{}}_part0.npy"
CHAINS = range(240, 270)
G = {"k1-4": slice(1, 5), "k5-7": slice(5, 8), "k8-16": slice(8, 17), "k17-64": slice(17, 65), "k65+": slice(65, None)}

print(f"GT residual (minus forcing), Re{RE} val 240-269, 1 unit; share per band | rel ||r||/||f|| | k17+ power per frame rel to dt1/64 o2")
for res in RESS:
    raw = np.load(D.format(res), mmap_mode="r")
    kinf = cheb_bins(res, "cpu")
    base = None
    for sub in (2, 1):
        for order in (2, 4):
            ns = NSVorticity(re=RE, time_order=order)
            p, rr, ff = np.zeros(res), 0.0, 0.0
            for j in CHAINS:
                w = torch.from_numpy(np.ascontiguousarray(raw[j, ::sub])).permute(1, 2, 0)[None].double()
                du, _ = ns.residual(w)
                r = du - ns.get_forcing(res, "cpu").double()
                p += band_power(r, kinf, res) / r.shape[-1]
                rr += float(r.pow(2).sum())
                ff += float(ns.get_forcing(res, "cpu").double().pow(2).sum()) * r.shape[-1]
            k17 = p[17:].sum()
            base = k17 if base is None else base
            sh = " ".join(f"{p[s].sum() / p[1:].sum():.3f}" for s in G.values())
            print(f"res{res} dt1/{128 // sub} o{order}  " + " ".join(G) + f"  {sh} | rel {np.sqrt(rr / ff):.4f} | k17+ x{k17 / base:.3f}", flush=True)
