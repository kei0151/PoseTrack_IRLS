#!/bin/bash
# compare_scene.sh SCENE
# baseline (master) と IRLS (feature) を比較する
set -e

SCENE=${1:-scene_090}
ROOT=/home/suzuki/PoseTrack_IRLS
NAS_BASE=/mnt/vmlqnap02/dataset/AIC2025track1/MTMC_Tracking_2024
EVAL_DIR=${NAS_BASE}/eval
SPLIT=test
RESULT_DIR=${ROOT}/result/track
COMPARE_DIR=/tmp/posetrack_compare

mkdir -p ${COMPARE_DIR}

eval_method() {
    local METHOD=$1
    local TRACK_FILE=$2
    local OUT_FILE=${COMPARE_DIR}/${SCENE}_${METHOD}.txt

    # submission形式に変換
    conda run -n aic24 python -c "
import numpy as np, os
data = np.loadtxt('${TRACK_FILE}')
out = data[:, :-1]
np.savetxt('${OUT_FILE}', out, fmt='%d %d %d %d %d %d %d %f %f')
print('${METHOD} submission saved:', out.shape)
"

    # カメラJSON
    python3 -c "
import json, os
scene_dir = '${NAS_BASE}/${SPLIT}/${SCENE}'
cams = sorted([d for d in os.listdir(scene_dir) if d.startswith('camera_')])
cam_ids = [int(c.replace('camera_0', '')) for c in cams]
d = [{'scene_name': '${SCENE}', 'camera_ids': cam_ids}]
os.makedirs('${ROOT}/eval', exist_ok=True)
json.dump(d, open('${ROOT}/eval/${SCENE}_cam.json', 'w'))
"

    echo ""
    echo "========== ${METHOD} =========="
    conda run -n eval311 python3 ${EVAL_DIR}/main.py \
        --prediction_file ${OUT_FILE} \
        --ground_truth_file ${NAS_BASE}/${SPLIT}/${SCENE}/ground_truth.txt \
        --num_cores 8 \
        --scene_2_camera_id_file ${ROOT}/eval/${SCENE}_cam.json \
        2>&1 | grep -E "HOTA:|DetA:|AssA:|LocA:|${SCENE}"
}

# --- baseline (master) ---
echo ">>> Switching to master for baseline..."
git stash
git checkout master
echo ">>> Running baseline tracking..."
conda run -n aic24 bash -c "
  cd ${ROOT}/track && python run_tracking_batch.py ${SCENE} 2>&1 | tee ${COMPARE_DIR}/${SCENE}_baseline_track.log
"
cp ${RESULT_DIR}/${SCENE}.txt ${COMPARE_DIR}/${SCENE}_baseline_raw.txt
eval_method "baseline" ${COMPARE_DIR}/${SCENE}_baseline_raw.txt

# --- IRLS (feature branch) ---
echo ">>> Switching to feature/irls-triangulation..."
git checkout feature/irls-triangulation
git stash pop 2>/dev/null || true
echo ">>> Running IRLS tracking..."
conda run -n aic24 bash -c "
  cd ${ROOT}/track && python run_tracking_batch.py ${SCENE} 2>&1 | tee ${COMPARE_DIR}/${SCENE}_irls_track.log
"
cp ${RESULT_DIR}/${SCENE}.txt ${COMPARE_DIR}/${SCENE}_irls_raw.txt
eval_method "irls" ${COMPARE_DIR}/${SCENE}_irls_raw.txt

echo ""
echo "=========================================="
echo " 比較完了: ${SCENE}"
echo " baseline: ${COMPARE_DIR}/${SCENE}_baseline_raw.txt"
echo " irls:     ${COMPARE_DIR}/${SCENE}_irls_raw.txt"
echo "=========================================="
