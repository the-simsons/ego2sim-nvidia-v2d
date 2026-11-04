"""Episodes of the GMD runs: S0 teacher checkpoints, S0 run directories (curves, overrides), world counts, reconstruction.
GMD_SET=v2 selects the second run (each episode's best compatible S0 checkpoint on its best reconstruction); default: first run.
GMD_SET=v5: run 2's table plus the new episodes of run 5 (23 episodes; fused ep1 / ep28 on v2clean2); an optional 5th field
is the S0 phase directory (default p1_96M).
For evaluation episodes the 4th field is the reconstruction (v1, v2recon, v2clean); for public episodes the parquet suffix."""
import os

EVAL = {1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38}
S0 = {  # episode: (S0 run dir under training_s0, best 64-world checkpoint, worlds, parquet suffix for public episodes)
    41: ("ep41_v4_rayreward5decay", 75497472, 2048, "_teammate"), 40: ("ep40_v4_rayreward5decay", 62914560, 2048, "_teammate"),
    26: ("ep26_v1_rayreward5decay", 96776192, 1024, "_teammate"), 18: ("ep18_v9_rayreward5decay", 92274688, 1024, ""),
    2: ("ep2_v1_rayreward5decay", 83886080, 1024, "_teammate"),
    38: ("eval/ep38_v1_cleanedref_ray5decay", 62914560, 2048, None), 37: ("eval/ep37_v1_cleanedref_ray5decay", 67108864, 2048, None),
    35: ("eval/ep35_v1_cleanedref_ray5decay", 71303168, 2048, None), 36: ("eval/ep36_v1_cleanedref_ray5decay", 88080384, 2048, None),
    6: ("eval/ep6_v1_cleanedref_ray5decay", 88080384, 1024, None), 8: ("eval/ep8_v1_cleanedref_ray5decay", 92274688, 1024, None),
    3: ("eval/ep3_v1_cleanedref_ray5decay", 96468992, 1024, None), 28: ("eval/ep28_v1_cleanedref_ray5decay", 12582912, 1024, None),
    19: ("eval/ep19_v1_cleanedref_ray5decay", 54525952, 1024, None), 5: ("eval/ep5_v1_cleanedref_ray5decay", 96776192, 1024, None),
}
S0_V2 = {  # best 64-world ADD-AUC checkpoint of a relative_hand_pose run (ep40_v1_s0inputs, 0.995, uses the residual action)
    41: ("ep41_v4_rayreward5decay", 75497472, 2048, "_teammate"), 40: ("ep40_v4_rayreward5decay", 62914560, 2048, "_teammate"),
    26: ("ep26_v1_rayreward5decay", 79691776, 1024, "_teammate"), 18: ("ep18_v9_rayreward5decay", 75497472, 1024, ""),
    2: ("ep2_v1_rayreward5decay", 46137344, 1024, "_teammate"),
    35: ("eval/ep35_v2recon_ray5decay", 83886080, 2048, "v2recon"), 8: ("eval/ep8_v2clean_ray5decay", 37748736, 1024, "v2clean"),
    3: ("eval/ep3_v2recon_ray5decay", 75497472, 1024, "v2recon"), 36: ("eval/ep36_v2clean_ray5decay", 67108864, 2048, "v2clean"),
    37: ("eval/ep37_v2recon_ray5decay", 71303168, 2048, "v2recon"), 6: ("eval/ep6_v2recon_ray5decay", 96776192, 1024, "v2recon"),
    15: ("eval/ep15_v2recon_ray5decay", 29360128, 1024, "v2recon"), 30: ("eval/ep30_v2recon_ray5decay", 54525952, 1024, "v2recon"),
    19: ("eval/ep19_v2recon_ray5decay", 41943040, 1024, "v2recon"), 28: ("eval/ep28_v2recon_ray5decay", 92274688, 1024, "v2recon"),
}
M_RUNS_V2 = {"A_seen": [41, 26, 3], "B_unseen": [37, 6, 15, 30], "C_s0_weak": [19, 28]}
S0_V5 = {k: v for k, v in S0_V2.items() if k != 28}
S0_V5.update({  # best 64-world ADD-AUC checkpoint of each new episode's S0 run on its best v2 reconstruction
    1: ("eval/ep1_v2clean2_ray5decay", 81788928, 1024, "v2clean2", "p1_81M"), 14: ("eval/ep14_v2clean2_ray5decay", 79691776, 1024, "v2clean2", "p1_81M"),
    20: ("eval/ep20_v2clean2_ray5decay", 92274688, 1024, "v2clean2"), 28: ("eval/ep28_v2clean2_ray5decay", 79691776, 1024, "v2clean2"),
    5: ("eval/ep5_v2recon_ray5decay", 62914560, 1024, "v2recon"), 38: ("eval/ep38_v2recon_ray5decay", 96468992, 2048, "v2recon"),
    16: ("eval/ep16_v2clean_ray5decay", 83886080, 1024, "v2clean"), 32: ("eval/ep32_v2clean_ray5decay", 46137344, 1024, "v2clean"),
    13: ("ep13_v1_rayreward5decay", 75497472, 1024, "_teammate"),
})
G_TRAIN_V5 = [41, 40, 26, 18, 2, 1, 3, 8, 14, 28, 35, 36]
M_RUNS_V5 = {"A_seen": G_TRAIN_V5, "B_unseen": [37, 38, 6, 15, 30, 32], "C_s0_weak": [5, 16, 19, 20, 13]}
MESHMIX_V5 = [14, 15, 16, 19, 20, 30, 38]     # episodes with a v1 mesh that differs from v2 (not fused): per-world mixing
if os.environ.get("GMD_SET") == "v2":
    S0 = S0_V2
elif os.environ.get("GMD_SET") == "v5":
    S0 = S0_V5
G_TRAIN = [41, 40, 26, 18, 2, 35, 8, 3, 36]
G_HELDOUT = [38, 6, 37]
M_RUNS = {"A_seen": [41, 26], "B_unseen_good": [38, 6, 37], "C_unseen_failed": [28, 19, 5]}


def ckpt(ep):
    d, s = S0[ep][:2]
    phase = S0[ep][4] if len(S0[ep]) > 4 else "p1_96M"
    return f"/ckpt/{d}/{phase}/policy_{s}.safetensors"


def parquet(ep):
    if ep in EVAL:
        return f"/pkg/sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave"
    return f"/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=episode_{ep:06d}{S0[ep][3]}/robot_name=sharpa_wave"


def recon(ep):
    """Reconstruction package of an evaluation episode (run_fc.sh RECON); 'public' for public episodes."""
    if ep not in EVAL:
        return "public"
    r = S0[ep][3] if ep in S0 else None
    return r if r in ("v2recon", "v2clean", "v2clean2") else "v1"


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[2]); f = sys.argv[1]
    print({"ckpt": lambda: ckpt(ep), "worlds": lambda: S0[ep][2], "parquet": lambda: parquet(ep), "s0dir": lambda: S0[ep][0],
           "recon": lambda: recon(ep)}[f]())
