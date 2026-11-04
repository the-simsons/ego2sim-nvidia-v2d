"""Render an episode's reconstructed scene (MANO human hands + our object meshes + table) to MP4 with Newton's headless GL
viewer, same look/camera as the FlashCHORD policy video (view_policy.py viewer.backend=mp4).
  render_recon_gl.py PARQUET MANO_NPZ OUT.mp4 [--eye x,y,z] [--target x,y,z] [--size 1280x720] [--table-z 0.75]
Run in the flash_chord venv with PYGLET_HEADLESS=true."""
import argparse
import numpy as np, pyarrow.parquet as pq, trimesh, warp as wp
from scipy.spatial.transform import Rotation as R
from flash_chord.visualization.viewer import ViewerConfig, configure_camera, create_viewer

OBJ_COLORS = [(0.93, 0.33, 0.30), (0.30, 0.75, 0.40), (0.30, 0.50, 0.90), (0.95, 0.75, 0.25)]  # = mask overlay order
SKIN, WOOD, GROUND = (0.90, 0.72, 0.60), (0.72, 0.50, 0.32), (0.55, 0.57, 0.62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("mano"); ap.add_argument("out")
    ap.add_argument("--eye", default="0,-0.95,1.38"); ap.add_argument("--target", default="0,0.05,0.86")
    ap.add_argument("--size", default="1280x720"); ap.add_argument("--table-z", type=float, default=0.75)
    ap.add_argument("--table-radius", type=float, default=0.42); ap.add_argument("--fps", type=float, default=20.0)
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    r = pq.read_table(a.parquet).to_pylist()[0]
    hands = np.load(a.mano)
    P = np.asarray(r["object_body_position"], float); Q = np.asarray(r["object_body_wxyz"], float)
    n = len(P)
    cfg = ViewerConfig(backend="mp4", output_path=a.out, num_frames=n, width=W, height=H, video_fps=a.fps,
                       camera_position=tuple(float(v) for v in a.eye.split(",")),
                       camera_target=tuple(float(v) for v in a.target.split(",")))
    viewer = create_viewer(cfg)
    viewer.set_model(None)
    configure_camera(viewer, cfg)
    ident = wp.array([wp.transform_identity()], dtype=wp.transform)
    one = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3)
    mat = wp.array([wp.vec4(0.6, 0.0, 0.0, 0.0)], dtype=wp.vec4)

    def put(name, V, F, color, update=False):
        viewer.log_mesh(name, wp.array(np.asarray(V, np.float32), dtype=wp.vec3),
                        wp.array(np.asarray(F, np.int32).reshape(-1), dtype=wp.int32), color=color)
        viewer.log_instances(name + "_i", name, ident, one, wp.array([wp.vec3(*color)], dtype=wp.vec3), mat)

    table = trimesh.creation.cylinder(radius=a.table_radius, height=0.02, sections=96)
    table.apply_translation([0, 0, a.table_z - 0.01])
    ground = trimesh.creation.box([30, 30, 0.01]); ground.apply_translation([0, 0, -0.005])
    objs = [trimesh.load(p, force="mesh") for p in r["object_mesh_paths"]]
    for i in range(n):
        viewer.begin_frame(i / a.fps)
        put("ground", ground.vertices, ground.faces, GROUND)
        put("table", table.vertices, table.faces, WOOD)
        for k, m in enumerate(objs):
            w, x, y, z = Q[i, k]
            V = m.vertices @ R.from_quat([x, y, z, w]).as_matrix().T + P[i, k]
            put(f"obj{k}", V, m.faces, OBJ_COLORS[k % 4])
        for side in ("left", "right"):
            if f"{side}_V" in hands and i < len(hands[f"{side}_V"]):
                put(f"hand_{side}", hands[f"{side}_V"][i], hands[f"{side}_F"], SKIN)
        viewer.end_frame()
    viewer.close()


if __name__ == "__main__":
    main()