"""
Script 1 (RealSense): Capture + YOLO + Depth + Point Cloud
===========================================================
Replaces the Middlebury dataset version with live RealSense capture.

Usage:
    conda activate foundation_stereo
    python script1_realsense.py --target cup --method foundation_stereo
    python script1_realsense.py --target cup --method realsense_depth
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs


# ── Config ──
FOUNDATION_DIR = os.path.abspath("./FoundationStereo")
CKPT = f"{FOUNDATION_DIR}/pretrained_models/23-51-11/model_best_bp2-001.pth"
OUT_DIR = "./object_depth_output"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════
# RealSense Camera
# ══════════════════════════════════════════════════

class RealSenseCamera:
    """
    Captures color + depth from an Intel RealSense D4xx camera.
    Can also capture stereo IR pairs for FoundationStereo.
    """

    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Enable depth stream (for Option A)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Enable left + right IR streams (for Option B with FoundationStereo)
        self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        self.config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        # Start pipeline
        self.profile = self.pipeline.start(self.config)

        # Get camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # Get depth scale (converts raw depth units to meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Get stereo baseline
        # The baseline is the distance between left and right IR sensors
        left_stream = self.profile.get_stream(rs.stream.infrared, 1)
        right_stream = self.profile.get_stream(rs.stream.infrared, 2)
        left_extrinsics = left_stream.get_extrinsics_to(right_stream)
        self.baseline = abs(left_extrinsics.translation[0])  # in meters

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Wait for auto-exposure to settle
        for _ in range(30):
            self.pipeline.wait_for_frames()

        print(f"RealSense initialized: {width}x{height}")
        print(f"  Depth scale: {self.depth_scale}")
        print(f"  Baseline: {self.baseline:.4f}m")
        print(f"  Focal length: {self.depth_intrinsics.fx:.1f}px")

    def capture(self):
        """
        Capture one frame from all streams.
        
        Returns:
            color_image: (H, W, 3) BGR uint8
            depth_image: (H, W) float32 in meters
            ir_left: (H, W) uint8 left infrared
            ir_right: (H, W) uint8 right infrared
        """
        frames = self.pipeline.wait_for_frames()

        # Align depth to color frame
        aligned = self.align.process(frames)

        # Color
        color_frame = aligned.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # Depth (aligned to color, converted to meters)
        depth_frame = aligned.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

        # Stereo IR
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())

        return color_image, depth_image, ir_left, ir_right

    def get_intrinsics_matrix(self):
        """Get 3x3 camera intrinsic matrix."""
        i = self.depth_intrinsics
        K = np.array([
            [i.fx, 0, i.ppx],
            [0, i.fy, i.ppy],
            [0, 0, 1]
        ])
        return K

    def make_intrinsic_file(self, out_path):
        """Save intrinsics in FoundationStereo K.txt format."""
        K = self.get_intrinsics_matrix()
        with open(out_path, "w") as f:
            f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
            f.write(f"{self.baseline:.6f}\n")
        return out_path

    def depth_to_pointcloud(self, depth, color):
        """
        Convert depth map to colored point cloud using camera intrinsics.
        
        Args:
            depth: (H, W) depth in meters
            color: (H, W, 3) BGR image
        
        Returns:
            Open3D point cloud
        """
        K = self.get_intrinsics_matrix()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        valid = (Z > 0.1) & (Z < 3.0)  # 10cm to 3m range
        points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
        colors_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)[valid] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        return pcd

    def stop(self):
        self.pipeline.stop()


# ══════════════════════════════════════════════════
# Option A: Use RealSense built-in depth
# ══════════════════════════════════════════════════

def pipeline_realsense_depth(cam, target_class):
    """
    Simple pipeline using RealSense's built-in depth sensor.
    Faster but lower quality depth.
    """
    from ultralytics import YOLO

    # Capture
    print("Capturing frame...")
    color, depth, _, _ = cam.capture()
    cv2.imwrite(os.path.join(OUT_DIR, "color.png"), color)
    np.save(os.path.join(OUT_DIR, "depth_meter.npy"), depth)

    # YOLO detection
    print("Running YOLO26n...")
    model = YOLO("yolo26n.pt")
    results = model(color, conf=0.5)

    target_box = None
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        if target_class.lower() in cls.lower():
            target_box = list(map(int, box.xyxy[0].tolist()))
            print(f"Found: {cls} at {target_box}")
            break

    if target_box is None:
        print(f"{target_class} not found. Available:")
        for box in results[0].boxes:
            print(f"  - {results[0].names[int(box.cls)]}")
        return

    # Build full point cloud from RealSense depth
    print("Building point cloud...")
    pcd = cam.depth_to_pointcloud(depth, color)

    # Denoise
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.03)

    o3d.io.write_point_cloud(os.path.join(OUT_DIR, "cloud_denoise.ply"), pcd)

    # Extract object using bbox (no scaling needed — same resolution)
    K = cam.get_intrinsics_matrix()
    points = np.asarray(pcd.points)
    x1, y1, x2, y2 = target_box

    Z = points[:, 2]
    u = (points[:, 0] * K[0, 0] / Z) + K[0, 2]
    v = (points[:, 1] * K[1, 1] / Z) + K[1, 2]

    mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2) & (Z > 0)

    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        obj_pc.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

    # Save everything Script 2 needs
    save_outputs(obj_pc, target_box, K, cam.baseline, scale=1.0)


# ══════════════════════════════════════════════════
# Option B: Use FoundationStereo for better depth
# ══════════════════════════════════════════════════

def pipeline_foundation_stereo(cam, target_class, scale=0.5):
    """
    Better pipeline: capture stereo IR pair, run FoundationStereo,
    get high-quality depth for grasping.
    """
    from ultralytics import YOLO

    # Capture
    print("Capturing frame...")
    color, _, ir_left, ir_right = cam.capture()

    # Convert IR to 3-channel for FoundationStereo (it expects RGB)
    ir_left_3ch = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2BGR)
    ir_right_3ch = cv2.cvtColor(ir_right, cv2.COLOR_GRAY2BGR)

    # Save images
    cv2.imwrite(os.path.join(OUT_DIR, "color.png"), color)
    cv2.imwrite(os.path.join(OUT_DIR, "left.png"), ir_left_3ch)
    cv2.imwrite(os.path.join(OUT_DIR, "right.png"), ir_right_3ch)

    # Create intrinsic file for FoundationStereo
    intrinsic_file = cam.make_intrinsic_file(os.path.join(OUT_DIR, "K.txt"))

    # YOLO detection on color image
    print("Running YOLO26n...")
    model = YOLO("yolo26n.pt")
    results = model(color, conf=0.5)

    target_box = None
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        if target_class.lower() in cls.lower():
            target_box = list(map(int, box.xyxy[0].tolist()))
            print(f"Found: {cls} at {target_box}")
            break

    if target_box is None:
        print(f"{target_class} not found. Available:")
        for box in results[0].boxes:
            print(f"  - {results[0].names[int(box.cls)]}")
        return

    # Run FoundationStereo
    cmd = [
        "python3", f"{FOUNDATION_DIR}/scripts/run_demo.py",
        "--left_file", os.path.join(OUT_DIR, "left.png"),
        "--right_file", os.path.join(OUT_DIR, "right.png"),
        "--ckpt_dir", CKPT,
        "--out_dir", OUT_DIR,
        "--intrinsic_file", intrinsic_file,
        "--scale", str(scale),
        "--get_pc", "1",
        "--denoise_cloud", "1",
    ]

    print(f"Running FoundationStereo...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR:\n{result.stderr}")
        return

    # Load denoised cloud and extract object
    cloud_path = os.path.join(OUT_DIR, "cloud_denoise.ply")
    full_pc = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(full_pc.points)

    K = cam.get_intrinsics_matrix()
    focal_s = K[0, 0] * scale
    cx_s = K[0, 2] * scale
    cy_s = K[1, 2] * scale

    x1, y1, x2, y2 = target_box
    x1_s, y1_s = int(x1 * scale), int(y1 * scale)
    x2_s, y2_s = int(x2 * scale), int(y2 * scale)

    Z = points[:, 2]
    u = (points[:, 0] * focal_s / Z) + cx_s
    v = (points[:, 1] * focal_s / Z) + cy_s

    mask = (u >= x1_s) & (u <= x2_s) & (v >= y1_s) & (v <= y2_s) & (Z > 0)

    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(points[mask])
    if full_pc.has_colors():
        obj_pc.colors = o3d.utility.Vector3dVector(np.asarray(full_pc.colors)[mask])

    save_outputs(obj_pc, target_box, K, cam.baseline, scale=scale)


# ══════════════════════════════════════════════════
# Save outputs for Script 2
# ══════════════════════════════════════════════════

def save_outputs(obj_pc, target_box, K, baseline, scale):
    """Save all files that Script 2 needs."""
    print(f"\nObject point cloud: {len(obj_pc.points)} points")
    o3d.io.write_point_cloud(os.path.join(OUT_DIR, "object_pc.ply"), obj_pc)
    np.save(os.path.join(OUT_DIR, "object_points.npy"), np.asarray(obj_pc.points))
    np.save(os.path.join(OUT_DIR, "target_bbox.npy"), target_box)

    calib_data = {
        "focal": K[0, 0] * scale,
        "cx": K[0, 2] * scale,
        "cy": K[1, 2] * scale,
        "baseline": baseline,
        "scale": scale,
    }
    np.save(os.path.join(OUT_DIR, "calib_scaled.npy"), calib_data)
    print(f"Outputs saved to {OUT_DIR}")


# ══════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="cup")
    parser.add_argument("--method", type=str, default="realsense_depth",
                        choices=["realsense_depth", "foundation_stereo"],
                        help="realsense_depth: use built-in depth. "
                             "foundation_stereo: use FoundationStereo for better quality.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale for FoundationStereo")
    args = parser.parse_args()

    cam = RealSenseCamera(width=1280, height=720)

    try:
        if args.method == "realsense_depth":
            print("\nUsing RealSense built-in depth")
            pipeline_realsense_depth(cam, args.target)
        else:
            print("\nUsing FoundationStereo for depth")
            pipeline_foundation_stereo(cam, args.target, scale=args.scale)
    finally:
        cam.stop()


if __name__ == "__main__":
    main()