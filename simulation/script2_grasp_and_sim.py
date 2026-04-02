"""
Script 2: AnyGrasp grasp detection + UR10 real robot execution
via RTDE Run in anygrasp conda environment

Usage:
    cd /home/ax0/Documents/research/Cook2LTL/simulation/anygrasp/anygrasp_sdk/grasp_detection
    python script2_grasp_real.py --target cup --robot_ip 192.168.1.10

"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import subprocess

# ── Config ──
SCRIPT_DIR = "/home/ax0/Documents/research/Cook2LTL/simulation/anygrasp/anygrasp_sdk/grasp_detection"
DATA_DIR = "/home/ax0/Documents/research/Cook2LTL/simulation/object_depth_output"
CHECKPOINT = os.path.join(SCRIPT_DIR, "log/checkpoint_detection.tar")

# Point Cloud Loading

def load_object_cloud(data_dir, target_bbox, calib):
    """
    Extract object point cloud from full scene cloud using YOLO bbox.
    """

    cloud_path = os.path.join(data_dir, "cloud_denoise.py")
    if not os.path.exists(cloud_path):
        cloud_path = os.path.join(data_dir, "cloud.ply")

    # subprocess.run([
    #     "/home/ax0/miniconda3/envs/foundation_stereo/bin/python", 
    #     "convert_script.py", 
    #     "input_2.4.2_file.npy", 
    #     "output_legacy_file.csv"    
    # ])

    full_pc = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(full_pc.points)
    colors = np.asarray(full_pc.colors) if full_pc.has_colors() else \
    np.ones_like(points) * 0.5

    scale = calib["scale"]
    x1, y1 = int(target_bbox[0] * scale), int(target_bbox[1]*scale)
    x2, y2 = int(target_bbox[2] * scale), int(target_bbox[3]*scale)

    focal, cx, cy = calib["focal"], calib["cx"], calib["cy"]

    Z = points[:, 2]
    u = (points[:, 0]* focal / Z) + cx
    v = (points[:, 1]* focal / Z) + cy
    
    mask = (u>= x1) & (u<=x2) & (v>= y1) & (v<=y2) * (Z>0)
    return points[mask], colors[mask]


# Anygrasp

def run_anygrasp(points, colors):
    "Run AnyGrasp grasp detection"

    from gsnet import AnyGrasp

    anygrasp = AnyGrasp(CHECKPOINT)
    anygrasp.load_net()

    margin = 0.05
    lims = [
        points[:, 0].min - margin, points[:, 0].max() + margin,
        points[:, 1].min - margin, points[:, 1].max() + margin,
        0.0, points[:, 2].max() + margin,
    ]
    
    gg, cloud = anygrasp.get_grasp(
        points.astype(np.float32),
        colors.astype(np.float32),
        lims = lims, 
        apply_object_mask = True,
        dense_grasp=False, 
        collision_detection = True,
    )

    if len(gg) == 0:
        return None, None, cloud
    
    gg = gg.nms().sort_by_score()
    print(f"Detected {len(gg)} grasps, top score: {gg[0].score:.4f}")
    return gg, gg[0], cloud

def visualize_grasps(gg, cloud, top_n=5):
    """Visualize top grasps on the point cloud"""
    grippers = gg[:top_n].to_open3d_geometry_list()
    trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cloud.transform(trans_mat)
    for g in grippers:
        g.transform(trans_mat)
    o3d.viasualization.draw_geometries([cloud, *grippers])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="cup")
    parser.add_argument("--robot_ip", type=str, required=True, help="UR10 IP address")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--speed", type=float, default=0.1, help="Robot speed (m/s)")
    args = parser.parse_args()


    #Load data from  Script 1

    print(f"Loading data from {args.data_dir}")
    bbox_path = os.path.join(args.data_dir, "target_bbox.npy")
    calib_path = os.path.join(args.data_dir, "calib_scaled.npy")

    if os.path.exists(bbox_path) and os.path.exists(calib_path):
        target_bbox = np.load(bbox_path)
        calib = np.load(calib_path, allow_pickle=True).item()
        obj_points, obj_colors = load_object_cloud(args.data_dir, target_bbox, calib)

    else: 
        cloud_path = os.path.join(args.data_dir, "cloud_denoise.ply")
        pc = o3d.io.read_poinnt_cloud(cloud_path)
        obj_points = np.asarray(pc.points).astype(np.float32)
        obj_colors = np.asarray(pc.colors).astype(np.float32) if pc.has_colors() else np.ones_like(obj_points) * 0.5

    

if __name__ == "__main__":
    main()