"""Render one episode of a policy with FlashCHORD's viewer under the evaluation composition (no VOC, no resets, frame 0),
one world, one video frame per control step (use viewer.video_fps=20 for real time). With $GMD_G set, the checkpoint is
an M policy and the frozen G runs in the loop (genv.GEnv); without it, a plain FlashCHORD policy (e.g. an S0 teacher).
$GMD_START=F starts at reference frame F. $GMD_GRIP=<grip params .npz> (with $GMD_G and an M checkpoint): the scripted
hand is teleported onto the grip at the start frame and then held / carried by grip_ctrl (grip test), M drives the rest.
With $GMD_SCRIPT=<engage .npy (T, H)> no policy is loaded: wrists track the human reference (g_only.make_tracker), G
fingers, engage from the file (G-alone replay); $GMD_SCRIPT=ref without $GMD_G tracks the reference wrists and fingers.
  GMD_G=<g file> python render_m.py evaluation.checkpoint=<M checkpoint> viewer.backend=mp4 viewer.output_path=... (view_policy args)
"""
from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, "/env/gmd")
import jax  # noqa: E402
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

vp = runpy.run_path("/fc/scripts/view_policy.py", run_name="view_policy_lib")
import rollout  # noqa: E402
from genv import GEnv  # noqa: E402


class MViewer(vp["ViewerApp"]):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
        from flash_chord.training.flash_sac.runner import resolve_jax_device
        ck = str(vp["resolve_path"](args.config.evaluation.checkpoint))
        with_g = bool(os.environ.get("GMD_G"))
        if with_g:
            rollout.WarpRLEnv = GEnv
        cfg, metadata, training, setup, env, genv, _ = rollout.build(ck, 1)
        start = int(os.environ.get("GMD_START", "0"))
        env.reset(frame_id=start)
        base = jnp.asarray(env.observation.numpy().reshape(1, env.observation_dim))
        if with_g:
            genv._reset_state(jnp.ones(1, bool))
            if os.environ.get("GMD_GRIP"):
                import grip_ctrl
                gparams = dict(np.load(os.environ["GMD_GRIP"]))
                gparams = {k: v for k, v in gparams.items() if k not in ("rel_p_t", "rel_R_t", "eng_t", "t_switch", "t_follow0")}
                gparams.update(t_grasp=np.array(start + 3), approach=np.array(0.02))
                grip_ctrl.place_hand(env, genv, gparams)
            _, _, _, ca, _ = genv._live()
            base = jnp.concatenate([base, genv._extra(ca, genv._eng)], 1)
        device = resolve_jax_device(training.device)
        self.obs = jax.device_put(base, device)
        script = os.environ.get("GMD_SCRIPT")
        if script:
            from g_only import make_tracker
            sides = list(env.action.layout.sides); H = len(sides); T = setup.reference.num_frames
            track, _ = make_tracker(env, setup.reference, sides, T, track_fingers=script == "ref")
            eng = None if script == "ref" else np.load(script) > 0.5
            starts = list(env.action.layout.hand_action_starts)

            def scripted(_actor, _obs, key):
                a = track()
                if eng is not None:
                    e = np.where(eng[min(int(env.timestep.numpy()[0]), T - 1)], 1.0, -1.0)
                    a = np.concatenate([np.stack([a[:, s:s + 6] for s in starts], 1), e[None, :, None]], -1).reshape(1, 7 * H)
                return jnp.asarray(a, jnp.float32), key
            self.actor, self.policy, self.key = None, scripted, jax.random.PRNGKey(0)
        else:
            self.actor, self.policy, self.key, _ = load_policy_for_inference(ck, training, self.obs, genv.action_dim, deterministic=True, device=device)
            if os.environ.get("GMD_GRIP"):
                self.policy = grip_ctrl.make_hybrid(env, genv, setup, self.policy, gparams)
        self.env, self.genv, self.scene, self.reference = env, genv, setup.scene, setup.reference
        self.model = setup.scene.model; self.state = env.state_0
        self.viewer.set_model(self.model)
        print(f"[render_m] {ck} ({'G in the loop' if with_g else 'plain policy'}), one episode, one frame per control step", flush=True)

    def step(self):
        if int(self.env.timestep.numpy()[0]) >= self.reference.num_frames - 1:
            return
        a, self.key = self.policy(self.actor, self.obs, self.key)
        self.obs = self.genv.step(a).observation
        self.state = self.env.state_0
        self.sim_time += self.env.frame_dt
        if os.environ.get("GMD_DEBUG"):                  # object error vs reference every 20 steps
            import numpy as np
            t = int(self.env.timestep.numpy()[0])
            if t % 20 == 0:
                ids = self.env.command.body_ids_w.numpy()
                bq = self.env.state_0.body_q.numpy()[ids, :3]
                ref = np.asarray(self.reference.object_body_pos_w())[min(t - 1, self.reference.num_frames - 1)]
                print(f"[dbg] t {t} object err cm {np.round(100 * np.linalg.norm(bq - ref, axis=-1), 1)} action |a| {float(np.abs(np.asarray(a)).mean()):.3f}", flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


import hydra  # noqa: E402


@hydra.main(version_base="1.3", config_path="/fc/src/flash_chord/configs", config_name="view_policy")
def main(cfg):
    viewer = vp["instantiate_typed"](cfg.viewer, vp["ViewerConfig"])
    MViewer.launch_configured(cfg, viewer, device=cfg.training.device)


main()
