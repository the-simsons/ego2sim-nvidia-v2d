import sys, numpy as np
# cam_for.py for a reference package: cam_for_pkg_traj.py <package> <episode> <back_d> <up> -> "cam" "target" num_frames(50 fps)
pkg, ep, back_d, up = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
d = np.load(f"{pkg}/trajectories/ep{ep}.npz", allow_pickle=True)
obj = d["object_position"].reshape(-1, 3).mean(0)
hands = np.concatenate([d["robot_left_wrist_position"], d["robot_right_wrist_position"]]).mean(0)
target = 0.5 * (obj + hands); target[2] = max(target[2], obj[2]) + 0.03
back = hands[:2] - obj[:2]; back /= max(np.linalg.norm(back), 1e-6)
cam = np.r_[target[:2] + back_d * back, target[2] + up]
print(",".join(f"{v:.3f}" for v in cam), ",".join(f"{v:.3f}" for v in target), int(len(d["object_position"]) * 2.5))
