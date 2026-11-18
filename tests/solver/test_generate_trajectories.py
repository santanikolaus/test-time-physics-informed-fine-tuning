import argparse

import numpy as np

from src.solver.generate_trajectories import generate


def _args(outdir, t_units):
    return argparse.Namespace(re=40.0, x_res=16, t_res=4, t_units=t_units, n_samples=1, start=3,
                              burnin=0.1, seed=42, device="cpu", outdir=str(outdir), outfile=None)


def test_longer_chain_extends_the_one_unit_chain(tmp_path):
    generate(_args(tmp_path, 1))
    generate(_args(tmp_path, 2))

    short = np.load(tmp_path / "NS_fine_Re40_T4_res16_part3.npy")
    long = np.load(tmp_path / "NS_fine_Re40_T4_U2_res16_part3.npy")

    assert long.shape == (9, 16, 16)
    np.testing.assert_array_equal(long[:5], short)
