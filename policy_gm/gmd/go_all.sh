#!/usr/bin/env bash
# G by itself on every episode (g_only.py), 4 containers at a time (2 per GPU). Logs: full/gonly/logs/.
cd /usr1/e2s/retarget_env/gmd
O=/usr1/e2s/v2d_challenge_data/outputs/gmd/full; mkdir -p $O/gonly/logs
job() {  # <slot> <ep> <setting>
  local GPU=$(( $1 % 2 )) EP=$2 S=$3
  [[ -f $O/gonly/ep$EP/$S.json ]] && return
  mkdir -p $O/gonly/ep$EP
  MEM=10g timeout 1500 ./run_fc.sh $GPU $EP go_${EP}_$S -- "GMD_G=/out/full/g/g9.npz python /env/gmd/g_only.py $(python3 episodes.py ckpt $EP) /out/full/gonly/ep$EP $S" > $O/gonly/logs/ep${EP}_$S.log 2>&1
}
export -f job; export O
for EP in 5 19 28 38 6 37 41 40 26 18 2 35 8 3 36; do
  for S in track_g_gt track_ref; do echo "$EP $S"; done
  [[ -f $O/ep$EP/engage_timeline.npy ]] && echo "$EP track_g_teach"
done | awk '{print NR-1, $0}' | xargs -P 4 -L 1 bash -c 'job $(( $0 % 4 )) $1 $2'
echo "all done $(date -u)" >> $O/gonly/logs/go_all.log
