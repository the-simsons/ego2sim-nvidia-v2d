import sys, numpy as np
ep, back_d, up = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
d = np.load(f"/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/episodes/episode_{ep:06d}/hand_trajectories.npz")
obj = d["object_position"].reshape(-1, 3).mean(0)
hands = np.concatenate([d["sharpa_left_wrist_position"], d["sharpa_right_wrist_position"]]).mean(0)
target = 0.5 * (obj + hands); target[2] = max(target[2], obj[2]) + 0.03
back = hands[:2] - obj[:2]; back /= max(np.linalg.norm(back), 1e-6)
cam = np.r_[target[:2] + back_d * back, target[2] + up]
print(",".join(f"{v:.3f}" for v in cam), ",".join(f"{v:.3f}" for v in target))
