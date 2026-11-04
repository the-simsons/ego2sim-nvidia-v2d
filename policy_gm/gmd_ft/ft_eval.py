"""Score checkpoints with FTEnv in the loop (64 worlds, FlashCHORD evaluation composition: no VOC, frame-0 reset,
deterministic policy), as gmd/eval_m.py.

  ft_eval.py watch <run_dir>          score every checkpoint of <run_dir>/run as it appears -> <run_dir>/evals.jsonl;
                                      exits when <run_dir>/DONE exists and every checkpoint is scored
  ft_eval.py once <checkpoint> <out.json>
Mode and frozen actors come from the FT_* variables (ftenv.py); FT_PERTURB is honoured (off unless set).
$FT_ENV=joint scores jenv.JEnv checkpoints at the unfreeze scale of their own step.
Each row: step, add_auc, mppe_cm, chord_sr + the environment's diagnostics (FTEnv: gate_frac, offset_rel_mean, cap_frac;
JEnv: finger_dev_mean, finger_dev_gt_0.1, unfreeze_scale).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/ft"); sys.path.insert(0, "/env/gmd")


def main():
    import jax
    from flash_chord.evaluation import harness as H
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    import os
    import rollout
    if os.environ.get("FT_ENV") == "joint":
        from jenv import JEnv as Env
    else:
        from ftenv import FTEnv as Env

    built = {}

    def score(path):
        if not built:
            rollout.WarpRLEnv = Env
            built["b"] = rollout.build(str(path), 64)
        cfg, metadata, training, setup, env, genv, thresholds = built["b"]
        env.reset(frame_id=0)
        if hasattr(genv, "_diag"):
            genv._diag = genv._diag * 0; genv._diag_n = 0
        if hasattr(genv, "_dev"):
            genv._dev = genv._dev * 0; genv._dev_n = 0
        if hasattr(genv, "set_eval_step"):                # JEnv: the unfreeze scale the checkpoint was trained with
            m = re.findall(r"policy_(\d+)", Path(path).name)
            genv.set_eval_step(int(m[0]) if m else None)
        device = resolve_jax_device(training.device)
        obs = jax.device_put(genv.eval_obs(), device)
        st, policy, key, _ = load_policy_for_inference(str(path), training, obs, genv.action_dim, deterministic=True, device=device)
        ro = record_object_pose_rollout(env, genv, policy, st, key, obs, reference=setup.reference,
                                        step_count=setup.reference.num_frames, start_frame=0, progress_interval=0)
        m = H.score_rollout(ro, thresholds)
        return {"add_auc": float(m.add_auc), "mppe_cm": float(m.mppe_cm), "chord_sr": float(m.chord_sr), **genv.ft_diag()}

    if sys.argv[1] == "once":
        row = {"checkpoint": sys.argv[2], **score(sys.argv[2])}
        Path(sys.argv[3]).write_text(json.dumps(row) + "\n")
        print(f"[ft_eval] {row}", flush=True)
        return
    run = Path(sys.argv[2]); ck = run / "run"; log = run / "evals.jsonl"
    done = {json.loads(l)["step"] for l in open(log)} if log.exists() else set()
    while True:
        cks = sorted((int(re.findall(r"policy_(\d+)", p.name)[0]), p) for p in ck.glob("policy_*.safetensors")) if ck.exists() else []
        todo = [(s, p) for s, p in cks if s not in done]
        if not todo:
            if (run / "DONE").exists():
                break
            time.sleep(15); continue
        step, path = todo[0]
        time.sleep(3)
        row = {"step": step, **score(path), "time": time.time()}
        done.add(step)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[ft_eval] {step / 1e6:.1f}M add_auc {row['add_auc']:.3f} " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("step", "add_auc", "time")), flush=True)
    print("[ft_eval] finished", flush=True)


if __name__ == "__main__":
    main()
