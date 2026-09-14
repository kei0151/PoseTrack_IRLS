#!/bin/bash
# eval_scene.sh SCENE
# tracking結果からsubmission生成 → HOTA評価
set -e

SCENE=${1:-scene_061}
ROOT=/home/suzuki/PoseTrack_IRLS
NAS_BASE=/mnt/vmlqnap02/dataset/AIC2025track1/MTMC_Tracking_2024
EVAL_DIR=${NAS_BASE}/eval
SPLIT=test

# submission生成
conda run -n aic24 python -c "
import numpy as np, os
data = np.loadtxt('${ROOT}/result/track/${SCENE}.txt')
out = data[:, :-1]
os.makedirs('${ROOT}/result/submission', exist_ok=True)
np.savetxt('${ROOT}/result/submission/${SCENE}.txt', out, fmt='%d %d %d %d %d %d %d %f %f')
print('submission saved:', out.shape)
"

# カメラJSON生成
python3 -c "
import json, os
scene_dir = '${NAS_BASE}/${SPLIT}/${SCENE}'
cams = sorted([d for d in os.listdir(scene_dir) if d.startswith('camera_')])
cam_ids = [int(c.replace('camera_0', '')) for c in cams]
d = [{'scene_name': '${SCENE}', 'camera_ids': cam_ids}]
os.makedirs('${ROOT}/eval', exist_ok=True)
json.dump(d, open('${ROOT}/eval/${SCENE}_cam.json', 'w'))
"

# HOTA評価
conda run -n eval311 python3 ${EVAL_DIR}/main.py \
    --prediction_file ${ROOT}/result/submission/${SCENE}.txt \
    --ground_truth_file ${NAS_BASE}/${SPLIT}/${SCENE}/ground_truth.txt \
    --num_cores 8 \
    --scene_2_camera_id_file ${ROOT}/eval/${SCENE}_cam.json \
    2>&1 | grep -E "HOTA:|DetA:|AssA:|LocA:|scene_"
