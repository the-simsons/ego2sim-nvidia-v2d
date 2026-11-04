"""Train with FTEnv in place of FlashCHORD's `WarpRLEnv` (gmd_ft), then run the usual train_logged.py -> train_flash_sac.py.

As gmd/train_m.py (GEnv) but with ftenv.FTEnv ($FT_ENV=joint: jenv.JEnv); env as there (GMD_G, GMD_VOC=ff, S0_VOC_GATE, S0_EMA, METRICS_JSONL) plus the
FT_* variables of ftenv.py. FlashCHORD, FlashSAC and the gmd/ code are untouched.
"""
import os
import runpy
import sys

sys.path.insert(0, "/ft"); sys.path.insert(0, "/env/gmd")
import flash_chord.training.environment as fenv  # noqa: E402
if os.environ.get("FT_ENV") == "joint":           # plan (a): joint training of M and the fingers (jenv.py)
    from jenv import JEnv as Env  # noqa: E402
else:
    from ftenv import FTEnv as Env  # noqa: E402

fenv.WarpRLEnv = Env
if os.environ.get("GMD_VOC") == "ff":
    sys.path.insert(0, "/env/voc"); sys.path.insert(0, "/env/s0")
    import s0_voc_gate  # noqa: E402
    import voc_ff  # noqa: E402
    s0_voc_gate.install = lambda: voc_ff.install_training()
    if not os.environ.get("S0_VOC_GATE"):
        voc_ff.install_training()
sys.argv = ["/env/train_logged.py"] + sys.argv[1:]
runpy.run_path("/env/train_logged.py", run_name="__main__")
