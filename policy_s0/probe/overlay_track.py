"""Shaded overlay of a tracked object on the egocentric video: the raw tracker pose (bundle object_to_camera) and the same
pose flipped 180 deg about the mesh's long axis, side by side, for checking orientation ambiguities by eye.
  overlay_track.py <bundle dir> <ego video> <out png> <frame> [<frame> ...]"""
import sys
import cv2
import numpy as np
import trimesh


def render(img, mesh, T, K, color):
    fx, fy, cx, cy = K
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    n = mesh.face_normals @ T[:3, :3].T
    tri = v[mesh.faces]; cen = tri.mean(1)
    facing = -(n * cen / np.linalg.norm(cen, axis=1, keepdims=True)).sum(1)
    out = img.copy(); order = np.argsort(-cen[:, 2])
    for i in order:
        if tri[i, :, 2].min() <= 0.01:
            continue
        uv = np.stack([fx * tri[i, :, 0] / tri[i, :, 2] + cx, fy * tri[i, :, 1] / tri[i, :, 2] + cy], 1).astype(np.int32)
        shade = 0.25 + 0.75 * max(facing[i], 0.0)
        cv2.fillConvexPoly(out, uv, tuple(int(c * shade) for c in color))
    return cv2.addWeighted(img, 0.45, out, 0.55, 0)


def main(bdir, video, out, frames):
    z = np.load(f"{bdir}/result.npz"); mesh = trimesh.load(f"{bdir}/mesh.obj", force="mesh", process=False)
    if len(mesh.faces) > 6000:
        mesh = mesh.simplify_quadric_decimation(face_count=6000)
    K = [float(x) for x in z["camera_intrinsics"]]
    ext = np.ptp(mesh.vertices, 0); ax = int(np.argmax(ext))                     # long axis of the mesh
    F = np.eye(4); F[:3, :3] = trimesh.transformations.rotation_matrix(np.pi, np.eye(3)[ax])[:3, :3]
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2; Fc = trimesh.transformations.translation_matrix(c) @ F @ trimesh.transformations.translation_matrix(-c)
    cap = cv2.VideoCapture(video); rows = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        T = z["object_to_camera_transform"][f].astype(np.float64)
        a = render(img, mesh, T, K, (60, 200, 60)); b = render(img, mesh, T @ Fc, K, (60, 60, 220))
        for im, t in ((img, f"video f{f}"), (a, "tracked"), (b, "flipped 180 about long axis")):
            cv2.putText(im, t, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        rows.append(np.hstack([cv2.resize(x, (480, 270)) for x in (img, a, b)]))
    cv2.imwrite(out, np.vstack(rows)); print(out, "long axis", "xyz"[ax], "valid", bool(z["object_is_valid"][frames].all()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]])
