#!/usr/bin/env bash
# AssemblyHands val split (option B, approved 31 Oct 2026): val annotations + both val ego-image archives.
# Downloads into data/assemblyhands/raw/ (one new dir), records size and sha256 to data/assemblyhands/raw/manifest.tsv.
set -uo pipefail
cd "$(dirname "$0")/.."
G=methods/.venv-tools/bin/gdown
R=data/assemblyhands/raw
mkdir -p $R
get() {  # id relpath
  local out=$R/$2
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then echo "have $2"; else
    $G "$1" -O "$out" --no-cookies 2>&1 | tail -2 || true
  fi
  if [ -s "$out" ] && ! head -c 200 "$out" | grep -qi "<html"; then
    printf "%s\t%s\t%s\t%s\n" "$2" "$1" "$(stat -c %s "$out")" "$(sha256sum "$out" | cut -d' ' -f1)" >> $R/manifest.tsv
    echo "OK $2 $(du -h "$out" | cut -f1)"
  else
    echo "FAILED $2 (id $1): quota/sign-in or HTML page"; rm -f "$out"
  fi
}
printf "file\tdrive_id\tbytes\tsha256\n" > $R/manifest.tsv
get 11BXLwPQ9hZYu_doEUKDFR-s4zqzzlo8J annotations/val/assemblyhands_val_ego_calib_v1-1.json
get 1-TNviWrqyf_0xBBiqYaGHkpVGuGiMSub annotations/val/assemblyhands_val_ego_data_v1-1.json
get 1vvdFRwOynYdJVRWnlSFy5y1OphH7umiU annotations/val/assemblyhands_val_joint_3d_v1-1.json
get 1upansaUPxxxofabteTTXltSOKlAMJwCA annotations/val/assemblyhands_val_exo_calib_v1-1.json
get 1phxNZvJwEGqrxEwrD_AENZWr-nuzyCXe annotations/skeleton.txt
get 1R2V1nhdPfNS9dKNNligqvwYtor0xfBM5 annotations/README.txt
get 1mryH9BQ7QQLqwAuhKMhDJ4-0bH1VWZGT ego_images_rectified/val/nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433.tar.gz
get 1jnNrtuvoYhEH0qQhu6GZ7VMAdyOl85cw ego_images_rectified/val/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345-008-001.tar.gz
echo DL_DONE
