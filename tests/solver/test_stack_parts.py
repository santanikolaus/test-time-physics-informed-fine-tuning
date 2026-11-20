import numpy as np
import pytest

from src.solver import stack_parts as sp


def _part(path, value, frames):
    np.save(path, np.full((frames, 2, 2), value, dtype=np.float32))


def test_val_chains_come_from_the_long_pattern_cut_to_the_part_length(tmp_path):
    for j in (238, 239):
        _part(tmp_path / f"res4_part{j}.npy", j, 3)
    _part(tmp_path / "res4_U10_part240.npy", 240, 11)

    paths = sp.chain_paths(tmp_path, "res{res}_part{j}.npy", "res{res}_U10_part{j}.npy", 1000, 4, range(238, 241))
    mm = sp.stack(paths, tmp_path / "out.npy")

    assert mm.shape == (3, 3, 2, 2)
    assert mm[:, 0, 0, 0].tolist() == [238, 239, 240]


def test_stack_refuses_to_overwrite_and_to_skip_a_missing_chain(tmp_path):
    _part(tmp_path / "p0.npy", 0, 3)
    (tmp_path / "out.npy").touch()
    with pytest.raises(SystemExit, match="exists"):
        sp.stack([tmp_path / "p0.npy"], tmp_path / "out.npy")
    with pytest.raises(SystemExit, match="missing 1"):
        sp.stack([tmp_path / "p0.npy", tmp_path / "p1.npy"], tmp_path / "new.npy")


def test_increment_std_pools_strided_one_step_increments():
    a = np.arange(5, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2))
    b = 3 * a

    assert sp.increment_std([a, b], sub_t=2) == pytest.approx(np.std([2, 2, 2, 2, 6, 6, 6, 6] * 2))
