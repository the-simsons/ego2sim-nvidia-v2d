import numpy as np
import pytest

from handbench import format as hf
from handbench import paths


def _meta():
    m = {k: None for k in hf.META_KEYS}
    m.update(method="t", fps=20.25)
    return m


def _tracks(n=4):
    a = hf.empty(n, np.arange(n) / 20.25)
    r = hf.row_index(1, "R")
    a["valid"][r], a["source"][r], a["conf"][r] = True, "observed", 0.9
    a["joints_cam"][r] = np.c_[np.zeros((21, 2)), np.full(21, 0.4)]
    return a


def test_roundtrip(tmp_path):
    a = _tracks()
    hf.save(tmp_path / "h.npz", a, _meta())
    h = hf.load(tmp_path / "h.npz")
    assert len(h) == 8 and h.n_frames == 4
    assert h.scored_mask().sum() == 1
    assert h.by_side("R", "valid").tolist() == [False, True, False, False]


@pytest.mark.parametrize("breaker, msg", [
    (lambda a: a.update(source=np.where(a["valid"], "none", a["source"])), "source"),
    (lambda a: a["joints_cam"].__setitem__((3, 0, 2), -1.0), "behind"),
    (lambda a: a.update(t=np.repeat(np.arange(4) / 30.0, 2)), "30 fps"),
    (lambda a: a.update(conf=a["conf"] + 2), "conf"),
    (lambda a: a.update(side=np.tile(np.array(["R", "L"]), 4)), "alternate"),
])
def test_validator_catches(breaker, msg):
    a = _tracks()
    breaker(a)
    errs = hf.validate_arrays(a, _meta())
    assert any(msg in e for e in errs), errs


@pytest.mark.skipif(not (paths.ROOT / "assets/mano_right.npz").exists(), reason="MANO npz not available")
def test_mano_consistency_check():
    from handbench.mano import Mano

    m = Mano()
    a = _tracks()
    r = hf.row_index(1, "R")
    a["transl"][r] = [0, 0, 0.4]
    _, J = m.forward(a["global_orient"][r], a["hand_pose"][r], a["betas"][r], a["transl"][r], True)
    a["joints_cam"][r] = J[0]
    assert hf.validate_arrays(a, _meta(), check_mano=True) == []
    a["joints_cam"][r] += 0.02
    assert any("reproduce" in e for e in hf.validate_arrays(a, _meta(), check_mano=True))


@pytest.mark.skipif(not (paths.ROOT / "assets/mano_right.npz").exists(), reason="MANO npz not available")
def test_refit_global_recovers_left_and_right():
    from handbench.fit import refit_global
    from handbench.mano import Mano

    m = Mano()
    rng = np.random.default_rng(1)
    for is_right in (True, False):
        go, hp, be = rng.normal(0, 1, 3), rng.normal(0, 0.3, 45), rng.normal(0, 1, 10)
        V, _ = m.forward(go, hp, be, [0.05, -0.02, 0.45], is_right, 1.1)
        go2, tr2, s2 = refit_global(m, hp, be, is_right, V[0])
        V2, _ = m.forward(go2, hp, be, tr2, is_right, s2)
        assert np.abs(V2 - V).max() < 1e-6
