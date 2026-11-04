"""Run a Python module (or script) unchanged and record its peak CUDA memory from torch.

  python tools/run_module_with_peak.py PEAK_JSON (-m MODULE | SCRIPT) [ARGS ...]
Runs it as __main__ (runpy, same process, so torch.cuda.max_memory_allocated() covers it) and then writes
PEAK_JSON = {"torch_max_allocated_mib", "torch_max_reserved_mib"} (0 if CUDA was never used). Exit status is the
module's. Used by run_e0_standalone.py so each upstream step's peak is recorded without editing upstream code.
"""
from __future__ import annotations

import json
import runpy
import sys


def main():
    out, args = sys.argv[1], sys.argv[2:]
    if args[0] == "-m":
        target, rest, run = args[1], args[2:], lambda t: runpy.run_module(t, run_name="__main__", alter_sys=True)
    else:
        target, rest, run = args[0], args[1:], lambda t: runpy.run_path(t, run_name="__main__")
    sys.argv = [target] + rest
    code = 0
    try:
        run(target)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        peak = {"torch_max_allocated_mib": 0.0, "torch_max_reserved_mib": 0.0}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            peak = {"torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                    "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20}
        with open(out, "w") as f:
            json.dump(peak, f)
    sys.exit(code)


if __name__ == "__main__":
    main()
