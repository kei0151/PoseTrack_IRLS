"""
visualize_scene.py  --  AIC2024 BEV + カメラbbox 合成動画

Usage:
  python visualize_scene.py \
    --track /tmp/posetrack_compare/scene_090_baseline_raw.txt \
    --scene_dir /mnt/vmlqnap02/dataset/AIC2025track1/MTMC_Tracking_2024/test/scene_090 \
    --output /tmp/vis_scene090.mp4 \
    [--frame_start 0] [--frame_end 300] [--fps 15]
    [--gt /path/to/ground_truth.txt]
"""

import argparse
import os
import cv2
import json
import numpy as np
from collections import defaultdict

# Track IDごとの色（最大100人）
np.random.seed(42)
ID_COLORS = [(int(c[0]), int(c[1]), int(c[2]))
             for c in np.random.randint(60, 255, (200, 3))]

def color_for(tid):
    return ID_COLORS[int(tid) % len(ID_COLORS)]


def load_track(path):
    """cols: cam_id track_id frame_id x y w h world_x world_y [conf]"""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis]
    return data


def build_index(data):
    """frame_id -> list of rows"""
    idx = defaultdict(list)
    for row in data:
        fid = int(row[2])
        idx[fid].append(row)
    return idx


def world_bounds(data, gt=None):
    xs = list(data[:, 7])
    ys = list(data[:, 8])
    if gt is not None and len(gt):
        xs += list(gt[:, 7])
        ys += list(gt[:, 8])
    pad = 2.0
    return min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad


def world_to_bev(wx, wy, xmin, xmax, ymin, ymax, bev_w, bev_h):
    px = int((wx - xmin) / (xmax - xmin) * bev_w)
    py = int((1 - (wy - ymin) / (ymax - ymin)) * bev_h)
    return np.clip(px, 0, bev_w-1), np.clip(py, 0, bev_h-1)


def draw_bev(frame_rows, gt_rows, xmin, xmax, ymin, ymax, bev_w, bev_h, frame_id):
    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # グリッド線
    for gx in np.arange(int(xmin), int(xmax)+1, 5):
        px, _ = world_to_bev(gx, ymin, xmin, xmax, ymin, ymax, bev_w, bev_h)
        cv2.line(bev, (px, 0), (px, bev_h), (40, 40, 40), 1)
    for gy in np.arange(int(ymin), int(ymax)+1, 5):
        _, py = world_to_bev(xmin, gy, xmin, xmax, ymin, ymax, bev_w, bev_h)
        cv2.line(bev, (0, py), (bev_w, py), (40, 40, 40), 1)

    # GT（白X）
    for row in gt_rows:
        wx, wy = row[7], row[8]
        tid = int(row[1])
        px, py = world_to_bev(wx, wy, xmin, xmax, ymin, ymax, bev_w, bev_h)
        r = 8
        cv2.line(bev, (px-r, py-r), (px+r, py+r), (200, 200, 200), 2)
        cv2.line(bev, (px+r, py-r), (px-r, py+r), (200, 200, 200), 2)
        cv2.putText(bev, f"G{tid}", (px+5, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # 予測（カラー円）
    for row in frame_rows:
        wx, wy = row[7], row[8]
        tid = int(row[1])
        px, py = world_to_bev(wx, wy, xmin, xmax, ymin, ymax, bev_w, bev_h)
        col = color_for(tid)
        cv2.circle(bev, (px, py), 9, col, -1, cv2.LINE_AA)
        cv2.circle(bev, (px, py), 9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(bev, str(tid), (px+6, py+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    cv2.putText(bev, f"BEV  frame={frame_id}  pred={len(frame_rows)}  gt={len(gt_rows)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(bev, (0,0), (bev_w-1, bev_h-1), (255,255,255), 1)
    return bev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output", default="/tmp/vis_scene.mp4")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end",   type=int, default=300)
    parser.add_argument("--fps",         type=int, default=15)
    parser.add_argument("--gt",          default=None)
    parser.add_argument("--cam_w",       type=int, default=640)
    parser.add_argument("--cam_h",       type=int, default=360)
    parser.add_argument("--bev_w",       type=int, default=800)
    parser.add_argument("--bev_h",       type=int, default=600)
    parser.add_argument("--num_cams",    type=int, default=6,
                        help="カメラ映像を表示する台数（多いほど重い）")
    parser.add_argument("--cam_cols",    type=int, default=4,
                        help="カメラグリッドの列数")
    args = parser.parse_args()

    # データ読み込み
    data = load_track(args.track)
    gt = load_track(args.gt) if args.gt else np.empty((0, 10))
    pred_idx = build_index(data)
    gt_idx   = build_index(gt)

    xmin, xmax, ymin, ymax = world_bounds(data, gt if len(gt) else None)

    # カメラ一覧
    cam_dirs = sorted([
        d for d in os.listdir(args.scene_dir)
        if d.startswith("camera_") and os.path.isdir(os.path.join(args.scene_dir, d))
    ])
    # 表示するカメラを均等サンプリング
    step = max(1, len(cam_dirs) // args.num_cams)
    display_cams = cam_dirs[::step][:args.num_cams]

    # cam_id（数値）→ cam_dir名 のマッピング
    cam_id_map = {}
    for d in cam_dirs:
        num = int(d.replace("camera_", ""))
        cam_id_map[num] = d

    # カメラbbox index: cam_id -> frame_id -> rows
    cam_bbox_idx = defaultdict(lambda: defaultdict(list))
    for row in data:
        cid = int(row[0])
        fid = int(row[2])
        cam_bbox_idx[cid][fid].append(row)

    # VideoCapture
    caps = {}
    for cam_name in display_cams:
        vpath = os.path.join(args.scene_dir, cam_name, "video.mp4")
        if os.path.exists(vpath):
            caps[cam_name] = cv2.VideoCapture(vpath)

    # キャンバスレイアウト: 左にBEV、右にカメラグリッド
    n_cams = len(display_cams)
    cam_cols = args.cam_cols
    cam_rows = (n_cams + cam_cols - 1) // cam_cols
    cam_panel_w = args.cam_w * cam_cols
    cam_panel_h = args.cam_h * cam_rows
    canvas_h = max(args.bev_h, cam_panel_h)
    canvas_w = args.bev_w + cam_panel_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {args.output}")

    print(f"[INFO] cameras={len(cam_dirs)}, display={n_cams}")
    print(f"[INFO] world bounds x=[{xmin:.1f},{xmax:.1f}] y=[{ymin:.1f},{ymax:.1f}]")
    print(f"[INFO] canvas={canvas_w}x{canvas_h}, frames={args.frame_start}-{args.frame_end}")

    for fid in range(args.frame_start, args.frame_end + 1):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # BEV
        bev = draw_bev(pred_idx[fid], gt_idx[fid],
                       xmin, xmax, ymin, ymax,
                       args.bev_w, args.bev_h, fid)
        canvas[:args.bev_h, :args.bev_w] = bev

        # カメラ映像
        for i, cam_name in enumerate(display_cams):
            col_i = i % cam_cols
            row_i = i // cam_cols
            x0 = args.bev_w + col_i * args.cam_w
            y0 = row_i * args.cam_h

            # フレーム取得
            img = None
            if cam_name in caps:
                cap = caps[cam_name]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, img = cap.read()
                if not ret:
                    img = None
            if img is None:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img = cv2.resize(img, (args.cam_w, args.cam_h))

            # bbox描画
            cam_num = int(cam_name.replace("camera_", ""))
            sx = args.cam_w / 1920.0
            sy = args.cam_h / 1080.0
            for row in cam_bbox_idx[cam_num][fid]:
                tid = int(row[1])
                x, y, w, h = int(row[3]*sx), int(row[4]*sy), int(row[5]*sx), int(row[6]*sy)
                col = color_for(tid)
                cv2.rectangle(img, (x, y), (x+w, y+h), col, 2, cv2.LINE_AA)
                cv2.putText(img, str(tid), (x+2, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            cv2.rectangle(img, (0,0), (args.cam_w-1, args.cam_h-1), (255,255,255), 1)
            cv2.putText(img, cam_name, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            canvas[y0:y0+args.cam_h, x0:x0+args.cam_w] = img

        writer.write(canvas)
        if fid % 100 == 0:
            print(f"  frame {fid}/{args.frame_end}")

    writer.release()
    for cap in caps.values():
        cap.release()
    print(f"[DONE] -> {args.output}")


if __name__ == "__main__":
    main()
