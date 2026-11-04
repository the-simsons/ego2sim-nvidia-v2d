"""Multi-episode training on one concatenated reference (grasp_multi/build_multi.py): the episodes are segments of one
reference, objects of other episodes parked off the table. Adds, without touching FlashCHORD or the S0 modules:

  SegmentTerminationConfig  (termination=segment)  FlashCHORD tracking termination + truncation at each segment's last frame
  S0MultiResetConfig        (reset=s0_multi)       S0 resets per segment: a segment is drawn uniformly, then a uniform frame
                                                   inside it, or (acquisition mode) its first frame / a pre-pickup frame
  S0MultiObservationConfig  (observation=policy_s0_multi)  S0 observation + object geometry near the hands: for every hand
                                                   point (wrist + 5 fingertips per hand) and object, the vector to the
                                                   nearest of the object's surface points (posed by the live object pose,
                                                   clipped to 0.3 m, zero for parked objects), + the active-object mask
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.lifecycle.termination import TrackingTermination, TrackingTerminationConfig
from s0_observation import S0Observation, S0ObservationConfig
from s0_reset import S0Reset, S0ResetConfig, _REG


# ---------------------------------------------------------------------------------------------- termination
@wp.kernel
def _truncate_segment_end(timestep: wp.array(dtype=wp.int32), last_flag: wp.array(dtype=wp.int32), n: int,
                          truncated: wp.array(dtype=wp.int32)):
    w = wp.tid()
    f = wp.clamp(timestep[w], 0, n - 1)
    if last_flag[f] != 0:
        truncated[w] = 1


@dataclass(frozen=True)
class SegmentTerminationConfig(TrackingTerminationConfig):
    extras_path: str = ""

    def build(self, scene, object_body_ids_w, object_ref_pos_w, object_ref_quat_w, num_objects, device=None, episode_step=None):
        base = TrackingTermination.build(scene, config=self, object_body_ids_w=object_body_ids_w, object_ref_pos_w=object_ref_pos_w,
                                         object_ref_quat_w=object_ref_quat_w, num_objects=num_objects, device=device,
                                         episode_step=episode_step)
        ex = np.load(self.extras_path, allow_pickle=True)
        n = base.last_frame + 1
        flag = np.zeros(n, np.int32)
        flag[np.clip(ex["segment_last"], 0, n - 1)] = 1
        print(f"[s0 multi] segment truncation at frames {ex['segment_last'].tolist()}", flush=True)
        return SegmentTermination(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                                  last_flag=wp.array(flag, dtype=wp.int32, device=device), n_frames=n)


@dataclass
class SegmentTermination(TrackingTermination):
    last_flag: wp.array = None
    n_frames: int = 0

    def evaluate(self, body_q, timestep):
        super().evaluate(body_q, timestep)
        wp.launch(_truncate_segment_end, dim=self.buffers.world_count, inputs=[timestep, self.last_flag, self.n_frames],
                  outputs=[self.truncated])


# ---------------------------------------------------------------------------------------------- reset
@wp.kernel
def _sample_multi(reset_mask: wp.array(dtype=wp.int32), voc_decay_steps: int, always_first_frame: int,
                  first_frame_probability: wp.array(dtype=wp.float32), first_frame_voc_threshold: float,
                  reset_finger_openness: float, seed: int, curriculum_voc_scale: wp.array(dtype=wp.float32),
                  seg_start: wp.array(dtype=wp.int32), seg_last: wp.array(dtype=wp.int32), n_seg: int,
                  pickup_frames: wp.array(dtype=wp.int32), pickup_seg: wp.array(dtype=wp.int32), n_pickup: int,
                  perturb_probability: float, perturb_pos: float, perturb_yaw: float, num_objects: int,
                  reset_count: wp.array(dtype=wp.int32), reset_frame: wp.array(dtype=wp.int32), finger_scale: wp.array(dtype=wp.float32),
                  steps_since_reset: wp.array(dtype=wp.int32), applied_voc_scale: wp.array(dtype=wp.float32),
                  perturb_flag: wp.array(dtype=wp.int32), perturb_xyyaw: wp.array(dtype=wp.vec3)):
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    rng = wp.rand_init(seed, world + reset_count[world] * 104729)
    s = wp.min(int(wp.randf(rng) * float(n_seg)), n_seg - 1)
    p = first_frame_probability[0]
    frame = seg_start[s]
    uniform = int(0)
    if always_first_frame == 0:
        span = seg_last[s] - seg_start[s]                       # never start on the last frame (truncated at once)
        frame = seg_start[s] + wp.min(int(wp.randf(rng) * float(span)), span - 1)
        uniform = 1
        if p > 0.1 + 1.0e-6:
            if wp.randf(rng) < p:
                uniform = 0
                frame = seg_start[s]
                if n_pickup > 0 and wp.randf(rng) < 0.5:
                    k = wp.min(int(wp.randf(rng) * float(n_pickup)), n_pickup - 1)
                    frame = pickup_frames[k]
        elif curriculum_voc_scale[0] < first_frame_voc_threshold:
            if wp.randf(rng) < p:
                frame = seg_start[s]
                uniform = 0
    reset_frame[world] = frame
    finger_scale[world] = wp.randf(rng) * reset_finger_openness
    steps_since_reset[world] = 0
    if voc_decay_steps == 0:
        applied_voc_scale[world] = curriculum_voc_scale[0]
    else:
        applied_voc_scale[world] = 1.0
    perturb_flag[world] = 0
    if uniform == 1 and wp.randf(rng) < perturb_probability:
        perturb_flag[world] = 1
        for o in range(num_objects):
            perturb_xyyaw[world * num_objects + o] = wp.vec3((2.0 * wp.randf(rng) - 1.0) * perturb_pos,
                                                             (2.0 * wp.randf(rng) - 1.0) * perturb_pos,
                                                             (2.0 * wp.randf(rng) - 1.0) * perturb_yaw)
    reset_count[world] += 1


@dataclass(frozen=True)
class S0MultiResetConfig(S0ResetConfig):
    def build(self, world_count: int, num_frames: int, device=None):
        return S0MultiReset.build(world_count, num_frames, config=self, device=device)


@dataclass
class S0MultiReset(S0Reset):
    seg_start: wp.array = None
    seg_last: wp.array = None
    n_seg: int = 0
    pickup_seg: wp.array = None

    @classmethod
    def build(cls, world_count, num_frames, config=None, device=None):
        base = S0Reset.build(world_count, num_frames, config=config, device=device)
        ex = np.load(config.extras_path, allow_pickle=True)
        st, la = ex["segment_start"].astype(np.int32), ex["segment_last"].astype(np.int32)
        self = cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                   seg_start=wp.array(st, dtype=wp.int32, device=device), seg_last=wp.array(la, dtype=wp.int32, device=device),
                   n_seg=len(st), pickup_seg=wp.zeros(1, dtype=wp.int32, device=device))
        _REG["policy"] = self                                   # the perturbation hook reads this
        print(f"[s0 multi] {len(st)} segments, starts {st.tolist()}", flush=True)
        return self

    def sample(self, reset_mask, curriculum_voc_scale):
        c = self.config
        wp.launch(_sample_multi, dim=self.world_count, inputs=[
            reset_mask, c.voc_decay_steps, int(c.always_reset_to_first_frame), self.first_frame_probability,
            c.first_frame_voc_threshold, c.reset_finger_openness, c.seed, curriculum_voc_scale, self.seg_start, self.seg_last,
            self.n_seg, self.pickup_frames, self.pickup_seg, self.n_pickup, c.perturb_probability, c.perturb_position_m,
            float(np.radians(c.perturb_yaw_deg)), self.num_objects],
            outputs=[self.reset_count, self.reset_frame, self.finger_scale, self.steps_since_reset, self.applied_voc_scale,
                     self.perturb_flag, self.perturb_xyyaw])


# ---------------------------------------------------------------------------------------------- observation
@wp.kernel
def _append_geometry(inner: wp.array(dtype=wp.float32), inner_dim: int, out_dim: int, body_q: wp.array(dtype=wp.transform),
                     bodies_per_world: int, timestep: wp.array(dtype=wp.int32), num_frames: int, num_objects: int,
                     object_body_ids: wp.array(dtype=wp.int32), point_body_ids: wp.array(dtype=wp.int32),
                     point_body_pos: wp.array(dtype=wp.vec3), n_points: int, surf: wp.array(dtype=wp.vec3), n_surf: int,
                     active: wp.array(dtype=wp.float32), clip: float, out: wp.array(dtype=wp.float32)):
    w = wp.tid()
    src = w * inner_dim
    dst = w * out_dim
    for i in range(inner_dim):
        out[dst + i] = inner[src + i]
    c = dst + inner_dim
    frame = wp.clamp(timestep[w], 0, num_frames - 1)
    base = w * bodies_per_world
    for j in range(n_points):
        p = wp.transform_point(body_q[base + point_body_ids[j]], point_body_pos[j])
        for o in range(num_objects):
            a = active[frame * num_objects + o]
            xf = body_q[base + object_body_ids[o]]
            pl = wp.transform_point(wp.transform_inverse(xf), p)
            best = float(1.0e9)
            bv = wp.vec3(0.0, 0.0, 0.0)
            for i in range(n_surf):
                dv = surf[o * n_surf + i] - pl
                d2 = wp.dot(dv, dv)
                if d2 < best:
                    best = d2
                    bv = dv
            v = wp.transform_vector(xf, bv)
            n = wp.length(v)
            if n > clip:
                v = v * (clip / n)
            out[c + 0] = v[0] * a
            out[c + 1] = v[1] * a
            out[c + 2] = v[2] * a
            c += 3
    for o in range(num_objects):
        out[c + o] = active[frame * num_objects + o]


@dataclass(frozen=True)
class S0MultiObservationConfig(S0ObservationConfig):
    proximity_clip_m: float = 0.3

    def build(self, scene, action, command, contact, timestep, episode_step, reference_joint_q_offset, device=None):
        return S0MultiObservation.build(self, scene, action, command, contact, timestep, episode_step, device=device)


@dataclass
class S0MultiObservation:
    s0: S0Observation
    config: S0MultiObservationConfig
    world_count: int
    arrays: dict
    observation_dim: int
    block_names: tuple
    block_ranges: tuple
    observation: wp.array

    @classmethod
    def build(cls, config, scene, action, command, contact, timestep, episode_step, device=None):
        s0 = S0Observation.build(config, scene, action, command, contact, timestep, episode_step, device=device)
        ex = np.load(config.extras_path, allow_pickle=True)
        B = s0.num_objects
        surf = ex["object_surface_points"].astype(np.float32)
        act = ex["active"].astype(np.float32)
        if surf.shape[0] != B or act.shape != (s0.num_frames, B):
            raise ValueError(f"extras geometry {surf.shape} / active {act.shape} do not match {B} objects x {s0.num_frames} frames")
        n_points = int(s0.arrays["ray_points"])
        arrays = dict(surf=wp.array(surf.reshape(-1, 3), dtype=wp.vec3, device=device), n_surf=int(surf.shape[1]),
                      active=wp.array(act.reshape(-1), dtype=wp.float32, device=device), n_points=n_points)
        names, ranges = list(s0.block_names), list(s0.block_ranges)
        cursor = s0.observation_dim
        for name, width in ((f"hand_object_surface_vectors_{n_points}x{B}", 3 * n_points * B), ("active_objects", B)):
            names.append(name); ranges.append((cursor, cursor + width)); cursor += width
        print(f"[s0 multi] observation {s0.observation_dim} + geometry {cursor - s0.observation_dim} = {cursor}", flush=True)
        return cls(s0=s0, config=config, world_count=s0.world_count, arrays=arrays, observation_dim=cursor,
                   block_names=tuple(names), block_ranges=tuple(ranges),
                   observation=wp.zeros(s0.world_count * cursor, dtype=wp.float32, device=device))

    def __getattr__(self, name):                                 # anything else the env asks for comes from the S0 observation
        if name == "s0":
            raise AttributeError(name)
        return getattr(self.s0, name)

    def compute(self, state):
        return self.compute_into(state, self.observation)

    def compute_into(self, state, output):
        inner = self.s0.compute(state)
        s, a = self.s0, self.arrays
        wp.launch(_append_geometry, dim=self.world_count, inputs=[
            inner, s.observation_dim, self.observation_dim, state.body_q, s.bodies_per_world, s.timestep, s.num_frames,
            s.num_objects, s.command.body_ids, s.arrays["ray_body_ids"], s.arrays["ray_body_pos"], a["n_points"], a["surf"],
            a["n_surf"], a["active"], float(self.config.proximity_clip_m)], outputs=[output])
        return output
