"""Gate before policy training: read the provenance record next to a training input and refuse non-video sources
for a final result (plans/RECONSTRUCTION_PLAN.md section 8).

    python tools/check_provenance.py PATH [--final] [--copy-to OUT_DIR]

PATH is a ManoSharpaData partition, a bundle dir, or any file under one; provenance.json is searched in it and up to
4 parents. Prints the mode (video | mocap | mixed | unlabelled). Exit 0 if allowed, 2 if refused. --copy-to writes
the record (plus the mode) into the run's output dir so every checkpoint carries it."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2s import provenance  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--final", action="store_true", help="a final (reportable) result: video only")
    ap.add_argument("--copy-to")
    a = ap.parse_args(argv)
    rec = provenance.read(a.path)
    try:
        mode = provenance.require_video(rec, a.final, what=a.path)
    except RuntimeError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if mode != "video":
        print(f"note: training on {mode} trajectories ({(rec or {}).get('mocap_used_for')}); a development result only",
              file=sys.stderr)
    if a.copy_to:
        out = dict(rec or {"trajectory_source": None})
        out.update({"training_mode": mode, "final": a.final, "input": str(Path(a.path).resolve())})
        Path(a.copy_to).mkdir(parents=True, exist_ok=True)
        json.dump(out, open(Path(a.copy_to) / provenance.FILENAME, "w"), indent=1)
    print(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
