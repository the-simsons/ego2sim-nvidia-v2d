"""Train the movement policy M with a frozen grasp policy G in the loop (FlashSAC, S0 recipe).

FlashCHORD is untouched: this launcher replaces `WarpRLEnv` (the JAX view of the simulator) with genv.GEnv and then runs
our usual train_logged.py (local metrics, EMA, gated VOC) -> train_flash_sac.py.   env: GMD_G = G weights.
GMD_MESHMIX = meshmix_prep.py spec: per-world v1/v2 meshes (meshmix.py; needs collision.ground=false).
GMD_VOC=ff: the gated VOC is the feedforward one (voc/voc_ff.py: no 0.1 s lag, same gate, gains and clamps).
"""
import os
import runpy
import sys

sys.path.insert(0, "/env/gmd")
import flash_chord.training.environment as fenv  # noqa: E402
from genv import GEnv  # noqa: E402
import meshmix  # noqa: E402

meshmix.install()                                           # per-world v1/v2 meshes when $GMD_MESHMIX is set (needs collision.ground=false)

fenv.WarpRLEnv = GEnv
if os.environ.get("GMD_VOC") == "ff":
    sys.path.insert(0, "/env/voc"); sys.path.insert(0, "/env/s0")
    import s0_voc_gate  # noqa: E402
    import voc_ff  # noqa: E402
    s0_voc_gate.install = lambda: voc_ff.install_training()   # train_logged.py calls this when S0_VOC_GATE is set
    if not os.environ.get("S0_VOC_GATE"):
        voc_ff.install_training()
sys.argv = ["/env/train_logged.py"] + sys.argv[1:]
runpy.run_path("/env/train_logged.py", run_name="__main__")
