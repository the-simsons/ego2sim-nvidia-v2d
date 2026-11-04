"""Render a teacher-wrist + G-finger rollout with FlashCHORD's own viewer (view_policy.py), same logic as rollout.py.
  GMD_G=<g file with engage_timeline> python view_g.py evaluation.checkpoint=<teacher> viewer.backend=mp4 ...
GMD_G unset -> plain teacher (identical to view_policy.py).
"""
from __future__ import annotations

import os
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.argv[0] = "/fc/scripts/view_policy.py"
vp = runpy.run_path("/fc/scripts/view_policy.py", run_name="view_policy_lib")

from gfeat import Featurizer  # noqa: E402
from rollout import mlp_forward  # noqa: E402

Base = vp["FlashSACPolicyViewer"]


class GViewer(Base):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        g = os.environ.get("GMD_G")
        if not g:
            return
        env = self.env
        gp = dict(np.load(g, allow_pickle=True))
        lay = env.action.layout; cl = env.contact.layout; sl = self.scene.layout
        fr = lambda f: dict(name=f.name, body_id=int(f.body_id), pos=list(f.body_to_frame_pos), quat=list(f.body_to_frame_quat_xyzw))  # noqa: E731
        meta = dict(sides=list(lay.sides), fps=float(self.reference.fps), object_body_ids=np.asarray(cl.object_body_ids),
                    hand_slot_starts=np.asarray(cl.hand_slot_starts), link_counts=np.asarray(cl.link_counts),
                    surface_points=np.load(str(args.config.objective.extras_path), allow_pickle=True)["object_surface_points"].astype(np.float64),
                    frames={s: [fr(sl.hand(s).palm_frame)] + [fr(f) for f in sl.hand(s).dp_frames] + [fr(f) for f in sl.hand(s).fingertip_frames] for s in lay.sides},
                    finger_q_ids={s: np.asarray(sl.hand(s).finger_q_ids) for s in lay.sides},
                    finger_dof_ids={s: np.asarray(sl.hand(s).finger_dof_ids) for s in lay.sides})
        feat = Featurizer(meta)
        W = env.world_count; Hn = len(lay.sides); nf = int(lay.finger_counts[0])
        nq, nd = env.model.joint_coord_count // W, env.model.joint_dof_count // W; Bw = env.model.body_count // W
        wrist_idx = np.stack([np.arange(s, s + 6) for s in lay.hand_action_starts])
        fing_idx = np.stack([np.arange(s + 6, s + 6 + n) for s, n in zip(lay.hand_action_starts, lay.finger_counts)])
        hist = np.zeros((3, W, Hn, 6), np.float32); prev_f = np.zeros((W, Hn, nf), np.float32)
        teacher = self.policy_action

        def wrapped(state, observation, k):
            a, k = teacher(state, observation, k)
            raw = np.asarray(a, np.float32).reshape(W, -1)
            ts = env.timestep.numpy()
            hist[2], hist[1] = hist[1].copy(), hist[0].copy(); hist[0] = np.clip(raw[:, wrist_idx], -1, 1)
            eng = gp["engage_timeline"][np.clip(ts, 0, len(gp["engage_timeline"]) - 1)]
            x = feat(env.state_0.body_q.numpy().reshape(W, Bw, 7), env.state_0.joint_q.numpy().reshape(W, nq),
                     env.state_0.joint_qd.numpy().reshape(W, nd), env.contact.contact_active.numpy().reshape(W, -1).astype(np.float64),
                     env.contact.contact_force_w.numpy().reshape(W, -1, 3).astype(np.float64), hist, prev_f, eng)
            g_out = mlp_forward(gp, x.reshape(W * Hn, -1)).reshape(W, Hn, nf)
            exe = np.clip(raw, -1, 1)
            for h in range(Hn):
                exe[:, fing_idx[h]] = g_out[:, h]
            prev_f[:] = exe[:, fing_idx]
            return jnp.asarray(exe), k

        self.policy_action = wrapped
        print(f"[view_g] fingers from G: {g}", flush=True)


import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.core.override_parser.overrides_parser import OverridesParser  # noqa: E402


@hydra.main(version_base="1.3", config_path="/fc/src/flash_chord/configs", config_name="view_policy")
def main(cfg):
    """view_policy.py's main with the G viewer (FlashSAC checkpoints only)."""
    requested = vp["instantiate_typed"](cfg.evaluation, vp["PolicyViewConfig"])
    metadata = vp["read_checkpoint_metadata"](vp["resolve_path"](requested.checkpoint))
    saved = vp["checkpoint_config"](metadata)
    if requested.use_checkpoint_config:
        keys = tuple(o.get_key_element() for o in OverridesParser.create().parse_overrides(HydraConfig.get().overrides.task))
        cfg = vp["checkpoint_evaluation_config"](cfg, saved, keys, use_evaluation_training_world_count=False)
    viewer = vp["instantiate_typed"](cfg.viewer, vp["ViewerConfig"])
    GViewer.launch_configured(cfg, viewer, device=cfg.training.device)


main()
