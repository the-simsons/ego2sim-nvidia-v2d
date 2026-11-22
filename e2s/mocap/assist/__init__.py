"""Mocap-assisted reconstruction pieces (plans/RECONSTRUCTION_PLAN.md P0.1). Everything here USES motion capture, so its
outputs are labelled trajectory_source=mocap (or mixed) and are for development training and for scoring only:

    sync      video <-> mocap time offset from a video motion signal (tools/box_image_motion.py) and the mocap speed
    camera    camera path in the mocap world from an object tracker (FoundationPose) + the mocap object, with the
              look-alike orientation picked by Viterbi, the sync refined, and the path refined against the mask outline
    table     the table plane from the mocap object resting at frame 0, written as a frame-0 depth image

Configuration comes from the clip manifest (bench/clips/<clip>.json), never from constants in the code."""
