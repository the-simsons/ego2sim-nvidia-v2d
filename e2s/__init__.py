"""Ego2Sim library code (plans/RECONSTRUCTION_PLAN.md).

  e2s.geometry     pose maths shared by reconstruction and scoring
  e2s.provenance   where a trajectory came from (video or motion capture); read by policy training
  e2s.mocap        motion-capture loading, sync and fitting. SCORING AND TRAINING ONLY: reconstruction code must not
                   import it (enforced by tests/test_firewall.py)
"""
