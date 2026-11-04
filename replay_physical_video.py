"""Physical replay with the virtual object controller (VOC, replay=physical default: k=400, contacts on, hands PD-driven by the
reference, every object driven toward its reference pose) rendered to MP4, logging per-frame hand-wrist and object error."""
import json, os, sys
sys.path.insert(0, "/fc/scripts/debug")
from hydra import compose, initialize_config_dir

import replay as replay_script
from flash_chord.configuration import instantiate_typed
from flash_chord.visualization.viewer import ViewerConfig


class Phys(replay_script.Replay):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.err, self.out = [], os.environ["REPLAY_JSON"]

    def step(self):
        f = self.frame
        super().step()
        r = self.runner.check_termination()
        self.err.append({"frame": f, "wrist_pos_m": r.max_wrist_pos, "object_pos_m": r.max_object_pos, "object_ori_rad": r.max_object_ori})
        if len(self.err) % 50 == 0 or self.frame == 0:
            json.dump(self.err, open(self.out, "w"))


if __name__ == "__main__":
    with initialize_config_dir(config_dir="/fc/src/flash_chord/configs", version_base="1.3"):
        cfg = compose(config_name="replay", overrides=sys.argv[1:])
    Phys.launch_configured(cfg, instantiate_typed(cfg.viewer, ViewerConfig), device=cfg.device)
