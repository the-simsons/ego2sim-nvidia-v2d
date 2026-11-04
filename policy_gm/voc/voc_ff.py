"""Lag-free virtual object controller (VOC): tracking with reference feedforward.

FlashCHORD's VOC (runtime/object_control.py) applies, per object root and physics substep,
    a = k (x_ref[t] - x) - d v,    alpha = k rotvec(q_ref[t] q^-1) - d w
with the target held at reference frame t for the whole control step and damping toward zero velocity. An object
following a moving reference therefore trails it by about d / k = 0.1 s (k = 400, d = 40), plus up to one frame from the
held target. This module keeps the same gains, clamps, gravity compensation and scale, and changes only the tracking law:
    target = interpolation of frame t-1 -> t across the substeps (frame t reached at the end of the step, which is what
             the evaluator and the replay compare against)
    a      = k (x_tgt - x) + d (v_ref - v) + a_ref           (v_ref, a_ref: finite differences of the reference)
    alpha  = k rotvec(q_tgt q^-1) + d (w_ref - w)
Graph-safe: the substep index is a device counter advanced by a one-thread kernel after every call.

  install(env_or_runner, frame_dt, substeps, accel=True)  replaces the apply_object_control name used by
  flash_chord.envs.base and flash_chord.runtime.replay. Objects with articulations fall back to the original controller.
  install_training()  for RL training / evaluation envs: registers each RLEnv's timestep just before its step graph is
  captured, and keeps the S0 contact / wrench gate (s0_voc_gate) when it is registered: assist = voc_scale * gate, and no
  gravity compensation in wrench mode, exactly as s0_voc_gate does. Call it instead of s0_voc_gate.install().
"""
from __future__ import annotations

import warp as wp

import flash_chord.envs.base as _base
import flash_chord.runtime.replay as _replay
from flash_chord.runtime.object_control import VOCParams, apply_object_control as _original
from flash_chord.utils.quat import quat_to_rotvec

_REG: dict = {}


@wp.func
def _ref_pos(ref_pos: wp.array(dtype=wp.vec3), frame: int, num_frames: int, num_bodies: int, body: int) -> wp.vec3:
    f = wp.clamp(frame, 0, num_frames - 1)
    return ref_pos[f * num_bodies + body]


@wp.func
def _ref_quat(ref_quat: wp.array(dtype=wp.quat), frame: int, num_frames: int, num_bodies: int, body: int) -> wp.quat:
    f = wp.clamp(frame, 0, num_frames - 1)
    return ref_quat[f * num_bodies + body]


@wp.kernel
def _ff_object_control(
    body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector), body_mass: wp.array(dtype=wp.float32),
    body_com: wp.array(dtype=wp.vec3), body_inertia: wp.array(dtype=wp.mat33), object_root_body_ids: wp.array(dtype=wp.int32),
    object_body_offsets: wp.array(dtype=wp.int32), object_body_ids: wp.array(dtype=wp.int32),
    voc_reference_body_ids: wp.array(dtype=wp.int32), root_offsets: wp.array(dtype=wp.vec3),
    ref_pos: wp.array(dtype=wp.vec3), ref_quat: wp.array(dtype=wp.quat), num_frames: int, num_bodies: int,
    timestep: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32), substeps: int, frame_dt: float, use_accel: int,
    k_lin: float, d_lin: float, k_ang: float, d_ang: float, max_force: float, max_torque: float,
    scale: wp.array(dtype=wp.float32), scale_count: int, world_count: int, object_count: int, bodies_per_world: int,
    gravity: float, mass_normalized_gains: bool, gate: wp.array(dtype=wp.float32), use_gate: int,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    objects_per_world = object_count // world_count
    world = i // objects_per_world
    object_id = i - world * objects_per_world
    s = scale[0]
    if scale_count > 1:
        s = scale[world]
    if use_gate != 0:
        s = s * gate[i]
    t = timestep[world]
    rb = voc_reference_body_ids[object_id]
    off = root_offsets[i]
    p0 = _ref_pos(ref_pos, t - 2, num_frames, num_bodies, rb) + off
    p1 = _ref_pos(ref_pos, t - 1, num_frames, num_bodies, rb) + off
    p2 = _ref_pos(ref_pos, t, num_frames, num_bodies, rb) + off
    q1 = _ref_quat(ref_quat, t - 1, num_frames, num_bodies, rb)
    q2 = _ref_quat(ref_quat, t, num_frames, num_bodies, rb)
    if wp.dot(q1, q2) < 0.0:
        q1 = -q1
    alpha = float(counter[0] % substeps + 1) / float(substeps)
    tgt_p = p1 + alpha * (p2 - p1)
    tgt_q = wp.quat_slerp(q1, q2, alpha)
    v_ref = (p2 - p1) / frame_dt
    a_ref = wp.vec3(0.0, 0.0, 0.0)
    if use_accel != 0:
        a_ref = (p2 - 2.0 * p1 + p0) / (frame_dt * frame_dt)
    w_ref = quat_to_rotvec(wp.mul(q2, wp.quat_inverse(q1))) / frame_dt

    bid = object_root_body_ids[i]
    xf = body_q[bid]
    pos = wp.transform_get_translation(xf)
    quat = wp.transform_get_rotation(xf)
    vel = body_qd[bid]
    lin_vel_com = wp.spatial_top(vel)
    ang_vel = wp.spatial_bottom(vel)
    lin_vel = lin_vel_com - wp.cross(ang_vel, wp.quat_rotate(quat, body_com[bid]))
    mass = float(0.0)  # noqa: UP018
    for body_index in range(object_body_offsets[object_id], object_body_offsets[object_id + 1]):
        mass += body_mass[world * bodies_per_world + object_body_ids[body_index]]
    inertia = body_inertia[bid]
    inertia_eff = (inertia[0, 0] + inertia[1, 1] + inertia[2, 2]) / 3.0
    force = k_lin * (tgt_p - pos) + d_lin * (v_ref - lin_vel) + a_ref
    torque = k_ang * quat_to_rotvec(wp.mul(tgt_q, wp.quat_inverse(quat))) + d_ang * (w_ref - ang_vel)
    if mass_normalized_gains:
        force = mass * force
        torque = inertia_eff * torque
    force = force + wp.vec3(0.0, 0.0, gravity * mass)
    force = wp.vec3(wp.clamp(s * force[0], -max_force, max_force), wp.clamp(s * force[1], -max_force, max_force),
                    wp.clamp(s * force[2], -max_force, max_force))
    torque = wp.vec3(wp.clamp(s * torque[0], -max_torque, max_torque), wp.clamp(s * torque[1], -max_torque, max_torque),
                     wp.clamp(s * torque[2], -max_torque, max_torque))
    body_f[bid] = wp.spatial_vector(force, torque)


@wp.kernel
def _advance(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


def ff_apply_object_control(model, state, control, command, scale, params=None):
    r = _REG
    if not r or command.layout.num_articulations > 0:
        return _original(model, state, control, command, scale, params)
    p = params or VOCParams()
    ref = command.reference
    n = command.voc_body_ids_w.shape[0]
    gravity, gate, use_gate = p.gravity, r["ones"], 0
    g = _gate_reg()
    if g is not None and command.world_count == g["world_count"]:      # S0 contact / wrench gate (as s0_voc_gate)
        import s0_voc_gate as sg
        wrench_mode = sg.MODE == "wrench" and g.get("wrench_score") is not None
        if wrench_mode:
            wp.launch(sg._update_gate_wrench, dim=n, inputs=[g["wrench_score"], sg.WRENCH_THRESHOLD, sg.WRENCH_HOLD_SUBSTEPS], outputs=[g["hold"], g["gate"]])
        else:
            wp.launch(sg._update_gate, dim=n, inputs=[g["contact"].contact_active, g["slots_per_world"], g["hand_slot_starts"], g["link_counts"],
                                                      g["num_hands"], command.layout.num_objects, sg.HOLD_SUBSTEPS], outputs=[g["hold"], g["gate"]])
        gravity, gate, use_gate = (0.0 if wrench_mode else p.gravity), g["gate"], 1
    wp.launch(_ff_object_control, dim=n, inputs=[
        state.body_q, state.body_qd, model.body_mass, model.body_com, model.body_inertia, command.voc_body_ids_w,
        command.voc_object_body_offsets, command.voc_object_body_ids, command.voc_reference_body_ids,
        command.object_root_position_offsets_w, ref.body_pos_w, ref.body_quat_w, ref.num_frames, ref.num_bodies,
        r["timestep"], r["counter"], r["substeps"], r["frame_dt"], int(r["accel"]),
        p.k_lin, p.d_lin, p.k_ang, p.d_ang, p.max_force, p.max_torque, scale, scale.shape[0], command.world_count, n,
        command.bodies_per_world, gravity, p.mass_normalized_gains, gate, use_gate], outputs=[state.body_f])
    wp.launch(_advance, dim=1, inputs=[r["counter"]])


def _gate_reg():
    import sys
    sg = sys.modules.get("s0_voc_gate")
    return sg._REG if sg is not None and sg._REG else None


def install(owner, frame_dt: float, substeps: int, accel: bool = True) -> None:
    """owner: the env / replay runner whose `timestep` array the controller reads."""
    dev = owner.timestep.device
    n = int(owner.timestep.shape[0]) * 64
    _REG.update(timestep=owner.timestep, counter=wp.zeros(1, dtype=wp.int32, device=dev),
                frame_dt=float(frame_dt), substeps=int(substeps), accel=accel, ones=wp.ones(n, dtype=wp.float32, device=dev))
    _base.apply_object_control = ff_apply_object_control
    _replay.apply_object_control = ff_apply_object_control
    print(f"[voc_ff] feedforward VOC installed (frame_dt {frame_dt:.4f} s, {substeps} substeps, accel {accel})", flush=True)


def install_training(accel: bool = True) -> None:
    """Feedforward VOC for every RLEnv built from here on (registered just before its step graph is captured)."""
    from flash_chord.envs.rl import RLEnv
    if getattr(RLEnv.capture_step, "_voc_ff", False):
        return
    orig = RLEnv.capture_step

    def capture_step(self):
        install(self, self.config.sim.frame_dt, self.config.sim.substeps, accel)
        return orig(self)
    capture_step._voc_ff = True
    RLEnv.capture_step = capture_step
    _base.apply_object_control = ff_apply_object_control
    print("[voc_ff] feedforward VOC for training installed (keeps the S0 gate when registered)", flush=True)
