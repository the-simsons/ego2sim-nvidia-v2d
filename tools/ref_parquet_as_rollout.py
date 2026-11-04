"""Evaluator-format trajectory parquet straight from a FlashCHORD reference (ManoSharpaData) parquet, achieved :=
reference, so the reconstruction can be scored against mocap without a training run (ref_as_rollout.py needs one).

  python ref_parquet_as_rollout.py REFERENCE_PARQUET TEMPLATE_EVAL_PARQUET OUT
TEMPLATE: any official evaluator parquet (schema + metadata keys); its rows are not used. One world, start_frame 0, one
step per reference frame (REF_FPS 20 in ego2sim_episode.sh), body order = object_body_names.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ref, tmpl, out = sys.argv[1:4]
    r = pq.read_table(ref, columns=["fps", "object_body_names", "object_body_position", "object_body_wxyz"]).to_pylist()[0]
    names = r["object_body_names"]
    P = np.asarray(r["object_body_position"], np.float32)  # (T, B, 3)
    Q = np.asarray(r["object_body_wxyz"], np.float32)  # (T, B, 4) wxyz
    T, B = P.shape[:2]
    schema = pq.read_schema(tmpl)
    step = np.repeat(np.arange(T, dtype=np.int32), B)
    body = np.tile(np.arange(B, dtype=np.int32), T)
    p, q = P.reshape(-1, 3), Q.reshape(-1, 4)
    cols = {
        "environment_id": np.zeros(T * B, np.int32), "step": step, "reference_frame": step,
        "time_s": step / float(r["fps"]), "object_id": body, "object_body_id": body,
        "object_body_name": pa.DictionaryArray.from_arrays(pa.array(body), pa.array(names)),
        "environment_valid": np.ones(T * B, bool),
    }
    for pre in ("achieved_", "reference_"):
        for k, c in enumerate("xyz"):
            cols[f"{pre}position_{c}"] = p[:, k]
        for k, c in enumerate("wxyz"):
            cols[f"{pre}quaternion_{c}"] = q[:, k]
    meta = dict(schema.metadata or {})
    meta.update({b"flash_chord.step_count": str(T).encode(), b"flash_chord.object_body_count": str(B).encode(),
                 b"flash_chord.reference_parquet": ref.encode(), b"flash_chord.start_frame": b"0",
                 b"flash_chord.world_count": b"1", b"flash_chord.checkpoint_sha256": b"reference-only"})
    t = pa.table({f.name: pa.array(cols[f.name], type=f.type) if f.name != "object_body_name" else cols[f.name]
                  for f in schema}, schema=schema.with_metadata(meta))
    pq.write_table(t, out)
    print(f"{out}: {T} steps x {B} bodies {names}")


if __name__ == "__main__":
    main()
