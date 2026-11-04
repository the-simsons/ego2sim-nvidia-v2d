"""Render one episode of a jenv.JEnv checkpoint (Exp 3 / Exp 4) with FlashCHORD's viewer under the evaluation composition
(no VOC, frame-0 reset, deterministic), one world, one video frame per control step. Copy of gmd/render_m.py reduced to the
JEnv path; the unfreeze scale is the one the checkpoint was trained with (as ft_eval.py).
  GMD_G=<g file> FT_ENV=joint FJ_UNFREEZE=... python render_ft.py evaluation.checkpoint=<checkpoint> viewer.backend=mp4 ...
"""
from __future__ import annotations

import re
import runpy
import sys

sys.path.insert(0, "/ft"); sys.path.insert(0, "/env/gmd")
import jax  # noqa: E402

vp = runpy.run_path("/fc/scripts/view_policy.py", run_name="view_policy_lib")
import rollout  # noqa: E402
from jenv import JEnv  # noqa: E402


class JViewer(vp["ViewerApp"]):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
        from flash_chord.training.flash_sac.runner import resolve_jax_device
        ck = str(vp["resolve_path"](args.config.evaluation.checkpoint))
        rollout.WarpRLEnv = JEnv
        cfg, metadata, training, setup, env, genv, _ = rollout.build(ck, 1)
        env.reset(frame_id=0)
        m = re.findall(r"policy_(\d+)", ck)
        genv.set_eval_step(int(m[-1]) if m else None)
        device = resolve_jax_device(training.device)
        self.obs = jax.device_put(genv.eval_obs(), device)
        self.actor, self.policy, self.key, _ = load_policy_for_inference(ck, training, self.obs, genv.action_dim, deterministic=True, device=device)
        self.env, self.genv, self.scene, self.reference = env, genv, setup.scene, setup.reference
        self.model = setup.scene.model; self.state = env.state_0
        self.viewer.set_model(self.model)
        print(f"[render_ft] {ck}: unfreeze scale {genv.scale():.2f}, one episode, one frame per control step", flush=True)

    def step(self):
        if int(self.env.timestep.numpy()[0]) >= self.reference.num_frames - 1:
            return
        a, self.key = self.policy(self.actor, self.obs, self.key)
        self.obs = self.genv.step(a).observation
        self.state = self.env.state_0
        self.sim_time += self.env.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


import hydra  # noqa: E402


@hydra.main(version_base="1.3", config_path="/fc/src/flash_chord/configs", config_name="view_policy")
def main(cfg):
    viewer = vp["instantiate_typed"](cfg.viewer, vp["ViewerConfig"])
    JViewer.launch_configured(cfg, viewer, device=cfg.training.device)


main()
