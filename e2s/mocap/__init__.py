"""Motion capture: loading, time sync, rigid-body fitting, and the mocap-assisted tools.

SCORING AND TRAINING ONLY (plans/RECONSTRUCTION_PLAN.md, section 1 and 6). Reconstruction code must not import this
package; tests/test_firewall.py enforces it. Anything here that produces a trajectory or a camera path labels its
output trajectory_source "mocap" (e2s.provenance).
"""
