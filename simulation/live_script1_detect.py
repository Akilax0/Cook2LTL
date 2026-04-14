"""
Script 1 (Live Camera): YOLO → FoundationStereo → Point Cloud
==============================================================
Captures stereo pair via OpenCV, runs YOLO26n for object detection,
runs FoundationStereo for depth, extracts object point cloud.

Works with any stereo camera that outputs side-by-side frames via OpenCV
(ZED, custom stereo rigs, etc.)

Usage:
    conda activate foundation_stereo
    
    # Live capture from camera
    python script1_camera.py --target cup --device 0

    # Or use pre-saved stereo images
    python script1_camera.py --target cup --left left.png --right right.png

    # Specify camera intrinsics
    python script1_camera.py --target cup --device 0 --fx 700 --fy 700 --cx 640 --cy 360 --baseline 0.12
"""

import os
import sys
import subprocess
import argparse
import re
import numpy as np
import cv2
import open3d as o3d


# ── Config ──
FOUNDATION_DIR = os.path.abspath("./FoundationStereo")
CKPT = f"{FOUNDATION_DIR}/pretrained_models/23-51-11/model_best_bp2-001.pth"
OUT_DIR = "./object_depth_output"


# ══════════════════════════════════════════════════
# STEP 1: Capture stereo pair
# ══════════════════════════════════════════════════

def capture_stereo_from_camera(device_id, width=2560, height=720):
    """
    Capture a stereo pair from a side-by-side stereo camera.
    
    Most stereo cameras (ZED, custom rigs) output a single wide frame
    with left and right images stitched horizontally.
    We split it down the middle.
    
    Args:
        device_id: /dev/video index (0, 1, 2, etc.)
        width: full frame width (2x single image width)
        height: frame height
    
    Returns:
        left, right: BGR images as numpy arrays
    """
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        # Try V4L2 backend
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera device {device_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Discard first few frames (auto-exposure settling)
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Failed to grab frame")
        sys.exit(1)

    print(f"Captured frame: {frame.shape}")

    # Split side-by-side into left and right
    h, w = frame.shape[:2]
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    print(f"Left:  {left.shape}")
    print(f"Right: {right.shape}")

    return left, right


def load_stereo_from_files(left_path, right_path):
    """Load stereo pair from saved image files."""
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    if left is None:
        print(f"ERROR: Cannot read {left_path}")
        sys.exit(1)
    if right is None:
        print(f"ERROR: Cannot read {right_path}")
        sys.exit(1)

    print(f"Left:  {left.shape}")
    print(f"Right: {right.shape}")

    return left, right


# ══════════════════════════════════════════════════
# STEP 2: YOLO detection
# ══════════════════════════════════════════════════

def detect_target(image, target_class, conf=0.5):
    """
    Run YOLO26n on the left image and find the target object.
    
    Args:
        image: BGR image (left stereo image)
        target_class: object class to look for (e.g. "cup")
        conf: confidence threshold
    
    Returns:
        target_box: [x1, y1, x2, y2] or None if not found
    """
    from ultralytics import YOLO

    print(f"\nRunning YOLO26n (looking for '{target_class}')...")
    model = YOLO("yolo26n.pt")
    results = model(image, conf=conf)

    target_box = None
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        if target_class.lower() in cls.lower():
            target_box = list(map(int, box.xyxy[0].tolist()))
            print(f"Found: {cls} at {target_box}")
            return target_box

    print(f"'{target_class}' not found. Detected objects:")
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        c = float(box.conf[0])
        print(f"  - {cls} ({c:.2f})")

    return None


# ══════════════════════════════════════════════════
# STEP 3: Build intrinsic file + run FoundationStereo
# ══════════════════════════════════════════════════

def make_intrinsic_file(fx, fy, cx, cy, baseline, out_path):
    """
    Create FoundationStereo K.txt file.
    Line 1: flattened 3x3 intrinsic matrix
    Line 2: baseline in meters
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    with open(out_path, "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
        f.write(f"{baseline:.6f}\n")

    print(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} baseline={baseline:.4f}m")
    return K


def run_foundation_stereo(left_path, right_path, intrinsic_path, out_dir, scale=0.5):
    """
    Run FoundationStereo via subprocess on the stereo pair.
    Produces: vis.png, depth_meter.npy, cloud.ply, cloud_denoise.ply
    """
    cmd = [
        "python3", f"{FOUNDATION_DIR}/scripts/run_demo.py",
        "--left_file", left_path,
        "--right_file", right_path,
        "--ckpt_dir", CKPT,
        "--out_dir", out_dir,
        "--intrinsic_file", intrinsic_path,
        "--scale", str(scale),
        "--get_pc", "1",
        "--denoise_cloud", "1",
    ]

    print(f"\nRunning FoundationStereo (scale={scale})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(f"ERROR:\n{result.stderr}")
        sys.exit(1)

    print("FoundationStereo complete.")


# ══════════════════════════════════════════════════
# STEP 4: Extract object point cloud using YOLO bbox
# ══════════════════════════════════════════════════

def extract_object_cloud(cloud_path, target_box, K, scale):
    """
    Load the full scene point cloud from FoundationStereo,
    reproject each 3D point to 2D, and keep only points
    inside the YOLO bounding box.
    
    Args:
        cloud_path: path to cloud_denoise.ply
        target_box: [x1, y1, x2, y2] from YOLO (full resolution)
        K: 3x3 intrinsic matrix (full resolution)
        scale: downscale factor used for FoundationStereo
    
    Returns:
        obj_pc: Open3D point cloud of just the target object
    """
    full_pc = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(full_pc.points)

    # Scale bbox and intrinsics to match downscaled cloud
    x1_s = int(target_box[0] * scale)
    y1_s = int(target_box[1] * scale)
    x2_s = int(target_box[2] * scale)
    y2_s = int(target_box[3] * scale)

    focal_s = K[0, 0] * scale
    cx_s = K[0, 2] * scale
    cy_s = K[1, 2] * scale

    # Reproject 3D → 2D and filter by bbox
    Z = points[:, 2]
    u = (points[:, 0] * focal_s / Z) + cx_s
    v = (points[:, 1] * focal_s / Z) + cy_s

    mask = (u >= x1_s) & (u <= x2_s) & (v >= y1_s) & (v <= y2_s) & (Z > 0)

    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(points[mask])
    if full_pc.has_colors():
        obj_pc.colors = o3d.utility.Vector3dVector(np.asarray(full_pc.colors)[mask])

    print(f"Extracted {len(obj_pc.points)} object points from {len(points)} total")
    return obj_pc


# ══════════════════════════════════════════════════
# STEP 5: Save outputs for Script 2
# ══════════════════════════════════════════════════

def save_outputs(obj_pc, target_box, K, baseline, scale, out_dir):
    """Save everything Script 2 (AnyGrasp + RTDE) needs."""
    os.makedirs(out_dir, exist_ok=True)

    # Object point cloud
    o3d.io.write_point_cloud(os.path.join(out_dir, "object_pc.ply"), obj_pc)
    np.save(os.path.join(out_dir, "object_points.npy"), np.asarray(obj_pc.points))

    # YOLO bbox
    np.save(os.path.join(out_dir, "target_bbox.npy"), target_box)

    # Scaled calibration for Script 2
    calib_data = {
        "focal": K[0, 0] * scale,
        "cx": K[0, 2] * scale,
        "cy": K[1, 2] * scale,
        "baseline": baseline,
        "scale": scale,
    }
    np.save(os.path.join(out_dir, "calib_scaled.npy"), calib_data)

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  object_pc.ply      — {len(obj_pc.points)} points")
    print(f"  object_points.npy  — numpy array for AnyGrasp")
    print(f"  target_bbox.npy    — YOLO bounding box")
    print(f"  calib_scaled.npy   — camera calibration")


# ══════════════════════════════════════════════════
# STEP 6: Visualization
# ══════════════════════════════════════════════════

def visualize_detection(left, target_box, target_class, out_dir):
    """Save an image showing the YOLO detection on the left image."""
    viz = left.copy()
    x1, y1, x2, y2 = target_box
    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(viz, target_class, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "detection.png"), viz)


def visualize_object_cloud(obj_pc, out_dir):
    """Render object point cloud to image using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(obj_pc.points)
    colors = np.asarray(obj_pc.colors) if obj_pc.has_colors() else None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(pts[:, 0], -pts[:, 1], c=colors, s=0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.axis("off")
    fig.savefig(os.path.join(out_dir, "object_pc_render.png"),
                dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved visualization to {out_dir}/object_pc_render.png")


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Script 1: YOLO + FoundationStereo + Point Cloud")

    # Input source — either camera or files
    parser.add_argument("--device", type=int, default=None, help="Camera device index (e.g. 0)")
    parser.add_argument("--left", type=str, default=None, help="Path to left image file")
    parser.add_argument("--right", type=str, default=None, help="Path to right image file")

    # Camera settings
    parser.add_argument("--width", type=int, default=2560, help="Full stereo frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")

    # Camera intrinsics (defaults are approximate ZED 2 at HD720)
    parser.add_argument("--fx", type=float, default=700.0, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=700.0, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=640.0, help="Principal point x")
    parser.add_argument("--cy", type=float, default=360.0, help="Principal point y")
    parser.add_argument("--baseline", type=float, default=0.12, help="Stereo baseline (meters)")

    # Pipeline settings
    parser.add_argument("--target", type=str, default="cup", help="Object class to detect")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--scale", type=float, default=0.5, help="FoundationStereo downscale factor")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)

    # Camera calibration file (alternative to specifying fx/fy/cx/cy)
    parser.add_argument("--zed_conf", type=str, default=None,
                        help="Path to ZED .conf calibration file (e.g. /usr/local/zed/settings/SN12345.conf)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load intrinsics from ZED conf file if provided ──
    if args.zed_conf:
        print(f"Reading intrinsics from {args.zed_conf}")
        args.fx, args.fy, args.cx, args.cy, args.baseline = parse_zed_conf(args.zed_conf)

    # ── Step 1: Get stereo pair ──
    print("\n[STEP 1] Capturing stereo pair...")
    if args.device is not None:
        left, right = capture_stereo_from_camera(args.device, args.width, args.height)
    elif args.left and args.right:
        left, right = load_stereo_from_files(args.left, args.right)
    else:
        print("ERROR: Specify --device for camera or --left/--right for image files")
        sys.exit(1)

    # Save raw stereo images
    cv2.imwrite(os.path.join(args.out_dir, "left.png"), left)
    cv2.imwrite(os.path.join(args.out_dir, "right.png"), right)

    # ── Step 2: YOLO detection ──
    print("\n[STEP 2] Object detection...")
    target_box = detect_target(left, args.target, conf=args.conf)
    if target_box is None:
        print("Target not found. Exiting.")
        sys.exit(1)

    visualize_detection(left, target_box, args.target, args.out_dir)

    # ── Step 3: FoundationStereo ──
    print("\n[STEP 3] Stereo depth estimation...")
    intrinsic_path = os.path.join(args.out_dir, "K.txt")
    K = make_intrinsic_file(args.fx, args.fy, args.cx, args.cy, args.baseline, intrinsic_path)

    run_foundation_stereo(
        os.path.join(args.out_dir, "left.png"),
        os.path.join(args.out_dir, "right.png"),
        intrinsic_path,
        args.out_dir,
        scale=args.scale,
    )

    # ── Step 4: Extract object cloud ──
    print("\n[STEP 4] Extracting object point cloud...")
    cloud_path = os.path.join(args.out_dir, "cloud_denoise.ply")
    if not os.path.exists(cloud_path):
        cloud_path = os.path.join(args.out_dir, "cloud.ply")

    obj_pc = extract_object_cloud(cloud_path, target_box, K, args.scale)

    if len(obj_pc.points) < 100:
        print(f"WARNING: Only {len(obj_pc.points)} points — may not be enough for AnyGrasp")

    # ── Step 5: Save for Script 2 ──
    print("\n[STEP 5] Saving outputs...")
    save_outputs(obj_pc, target_box, K, args.baseline, args.scale, args.out_dir)

    # ── Step 6: Visualize ──
    visualize_object_cloud(obj_pc, args.out_dir)

    print("\n" + "=" * 50)
    print("Script 1 complete!")
    print(f"Run Script 2 next (in anygrasp env):")
    print(f"  python script2_grasp_real.py --target {args.target} --robot_ip <IP>")
    print("=" * 50)


def parse_zed_conf(conf_path):
    """
    Parse ZED camera .conf calibration file.
    These are stored at /usr/local/zed/settings/SN<serial>.conf
    """
    fx = fy = cx = cy = 0
    baseline = 0.12  # default ZED 2

    with open(conf_path) as f:
        section = ""
        for line in f:
            line = line.strip()
            if line.startswith("["):
                section = line