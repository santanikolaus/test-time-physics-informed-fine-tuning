import pytest
from omegaconf import OmegaConf

from msc.tta.adapt import adapt


def _cfg(heldout_split):
    return OmegaConf.create({
        "exp": "unet_nu100", "target_re": 500, "heldout_split": heldout_split,
        "objective": {"name": "physics", "pool_n": 5},
        "locus": {"name": "full", "patterns": ["*"], "layouts": {},
                  "shells": None, "t_modes": None},
        "lr": 2.0e-4, "steps": 500, "lr_milestones": [], "pde_re": 100,
    })


def _train_cfg():
    return OmegaConf.create({"data": {"data_path": "unused.npy"}})


@pytest.mark.parametrize("split, expected_offset", [("val", 240), ("test", 270)])
def test_build_splits_probes_the_requested_split(monkeypatch, split, expected_offset):
    built = []
    monkeypatch.setattr(adapt.setup, "data_path_for_re", lambda re: f"Re{re}.npy")
    monkeypatch.setattr(adapt.setup, "build_dataset",
                        lambda cfg, name: built.append(name) or list(range(240)))

    _, heldout, target_cfg = adapt.build_splits(_cfg(split), _train_cfg())

    assert built == [split, "train"]
    assert adapt.setup.SPLIT[split]["offset"] == expected_offset
    assert target_cfg.data.data_path == "Re500.npy"
    assert heldout is not None


def test_pool_path_swaps_only_the_pool_file(monkeypatch):
    paths = {}
    monkeypatch.setattr(adapt.setup, "data_path_for_re", lambda re: f"Re{re}.npy")
    monkeypatch.setattr(adapt.setup, "build_dataset",
                        lambda cfg, name: paths.setdefault(name, cfg.data.data_path) and list(range(15)))
    cfg = _cfg("test")
    cfg.pool_path = "/data/pool_tau2_res128.npy"

    _, _, target_cfg = adapt.build_splits(cfg, _train_cfg())

    assert paths == {"test": "Re500.npy", "train": "/data/pool_tau2_res128.npy"}
    assert target_cfg.data.data_path == "Re500.npy"
    assert "-pool_tau2_res128-" in adapt.run_name(cfg)


def test_target_path_swaps_pool_and_heldout_file(monkeypatch):
    paths = {}
    monkeypatch.setattr(adapt.setup, "data_path_for_re", lambda re: f"Re{re}.npy")
    monkeypatch.setattr(adapt.setup, "build_dataset",
                        lambda cfg, name: paths.setdefault(name, cfg.data.data_path) and list(range(15)))
    cfg = _cfg("val")
    cfg.target_path = "/data/Re500_res256.npy"

    _, _, target_cfg = adapt.build_splits(cfg, _train_cfg())

    assert paths == {"val": "/data/Re500_res256.npy", "train": "/data/Re500_res256.npy"}
    assert target_cfg.data.data_path == "/data/Re500_res256.npy"
    assert "-Re500_res256-" in adapt.run_name(cfg)


def test_pool_path_rejects_a_coarse_operator(monkeypatch):
    monkeypatch.setattr(adapt.setup, "data_path_for_re", lambda re: f"Re{re}.npy")
    monkeypatch.setattr(adapt.setup, "build_dataset", lambda cfg, name: list(range(15)))
    cfg = _cfg("test")
    cfg.pool_path = "pool.npy"
    train_cfg = OmegaConf.create({"data": {"data_path": "unused.npy", "coarse_path": "c.npy"}})
    with pytest.raises(ValueError, match="coarse_path"):
        adapt.build_splits(cfg, train_cfg)


def test_build_splits_rejects_the_train_split_as_probe_set():
    with pytest.raises(ValueError, match="heldout_split must be one of"):
        adapt.build_splits(_cfg("train"), _train_cfg())


def test_run_name_marks_the_locked_read_and_leaves_val_unmarked():
    assert adapt.run_name(_cfg("val")).endswith("-s500")
    assert adapt.run_name(_cfg("test")).endswith("-s500-test")


def test_run_name_marks_the_energy_term_only_when_on():
    cfg = _cfg("val")
    assert "-sp" not in adapt.run_name(cfg)
    cfg.spec_weight = 0.5
    assert "-sp0.5-" in adapt.run_name(cfg)
    cfg.spec_k_hi = 32
    assert "-sp0.5h32-" in adapt.run_name(cfg)


def test_run_name_marks_a_seed_and_unseeded_stays_unmarked():
    cfg = _cfg("val")
    assert "-seed" not in adapt.run_name(cfg)
    cfg.seed = 3
    assert adapt.run_name(cfg).endswith("-s500-seed3")


def test_load_config_defaults_to_unseeded():
    assert adapt.load_config(["experiment=fno"]).seed is None
