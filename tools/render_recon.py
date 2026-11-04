"""Render an episode's reconstructed 3D scene (MANO human hands + our object meshes + table) from the reference parquet
to MP4, from the same viewpoint as the FlashCHORD policy video (sim world frame, z up, table top at TABLE_Z).
  render_recon.py PARQUET OUT.mp4 [--eye x,y,z] [--target x,y,z] [--size 1280x720] [--table-z 0.75]
Run in the v2d_hamer env (manotorch, pyrender, pyarrow<18)."""
import argparse, os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, pyarrow.parquet as pq, pyrender, torch, trimesh, imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
from manotorch.manolayer import ManoLayer

OBJ_COLORS = [(0.93, 0.33, 0.30), (0.30, 0.75, 0.40), (0.30, 0.50, 0.90), (0.95, 0.75, 0.25)]  # match mask overlay order
SKIN = (0.88, 0.70, 0.58)


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    T = np.eye(4); T[:3, :3] = np.column_stack([s, u, -f]); T[:3, 3] = eye
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("out")
    ap.add_argument("--eye", default="0,-0.75,1.25"); ap.add_argument("--target", default="0,0,0.8")
    ap.add_argument("--size", default="1280x720"); ap.add_argument("--table-z", type=float, default=0.75)
    ap.add_argument("--table-radius", type=float, default=0.42); ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--yfov", type=float, default=45.0)
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    r = pq.read_table(a.parquet).to_pylist()[0]
    n = len(r["object_body_position"])
    # hands: MANO (flat hand mean, no centring) + per-frame shift onto the stored joints
    hands = {}
    for side in ("left", "right"):
        if not r.get(f"mano_{side}_trans"):
            continue
        m = ManoLayer(rot_mode="axisang", use_pca=False, side=side, center_idx=None, flat_hand_mean=True,
                      mano_assets_root=os.path.expanduser("~/v2d/mano"))
        go = torch.tensor(r[f"mano_{side}_global_orient"], dtype=torch.float32)
        fp = torch.tensor(r[f"mano_{side}_finger_pose"], dtype=torch.float32)
        b = torch.tensor(r[f"mano_{side}_betas"], dtype=torch.float32)[None].repeat(len(go), 1)
        with torch.no_grad():
            out = m(torch.cat([go, fp], 1), b)
        tr = np.asarray(r[f"mano_{side}_trans"], float)[:, None]
        V = out.verts.numpy() + tr; J = out.joints.numpy() + tr
        Js = np.asarray(r[f"mano_{side}_joints"], float)
        V = V + (Js - J).mean(1, keepdims=True)
        hands[side] = (V, m.th_faces.numpy())
    scene = pyrender.Scene(bg_color=[0.93, 0.94, 0.96, 1.0], ambient_light=[0.35, 0.35, 0.35])
    table = trimesh.creation.cylinder(radius=a.table_radius, height=0.02, sections=96)
    table.apply_translation([0, 0, a.table_z - 0.01])
    scene.add(pyrender.Mesh.from_trimesh(table, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.72, 0.50, 0.32, 1.0], roughnessFactor=0.8)))
    floor = trimesh.creation.box([3, 3, 0.01]); floor.apply_translation([0, 0, -0.005])
    scene.add(pyrender.Mesh.from_trimesh(floor, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.80, 0.81, 0.84, 1.0], roughnessFactor=1.0)))
    onodes = []
    for k, path in enumerate(r["object_mesh_paths"]):
        mesh = trimesh.load(path, force="mesh")
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*OBJ_COLORS[k % 4], 1.0], roughnessFactor=0.6)
        onodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False)))
    P = np.asarray(r["object_body_position"], float); Q = np.asarray(r["object_body_wxyz"], float)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(a.yfov), aspectRatio=W / H)
    eye = [float(v) for v in a.eye.split(",")]; tgt = [float(v) for v in a.target.split(",")]
    scene.add(cam, pose=look_at(eye, tgt))
    key = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(key, pose=look_at([0.6, -0.8, 2.0], [0, 0, 0.75]))
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.2), pose=look_at([-0.8, 0.4, 1.6], [0, 0, 0.75]))
    renderer = pyrender.OffscreenRenderer(W, H)
    hmat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*SKIN, 1.0], roughnessFactor=0.7)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    hnodes = []
    for i in range(n):
        for k, node in enumerate(onodes):
            T = np.eye(4); w, x, y, z = Q[i, k]; T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix(); T[:3, 3] = P[i, k]
            scene.set_pose(node, T)
        for node in hnodes:
            scene.remove_node(node)
        hnodes = []
        for side, (V, F) in hands.items():
            if i < len(V):
                tm = trimesh.Trimesh(V[i], F, process=False)
                hnodes.append(scene.add(pyrender.Mesh.from_trimesh(tm, material=hmat)))
        color, _ = renderer.render(scene)
        writer.append_data(color)
    writer.close(); renderer.delete()
    print(f"{a.out}: {n} frames {W}x{H}, hands {list(hands)}, objects {r['object_body_names']}")


if __name__ == "__main__":
    main()