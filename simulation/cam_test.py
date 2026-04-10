"""
ZED Camera Test Script
======================
Tests all ZED camera capabilities relevant to the grasping pipeline:
- Color capture
- Depth capture (built-in)
- Stereo pair (left + right RGB for FoundationStereo)
- Point cloud generation
- Camera intrinsics

Prerequisites:
    1. Install ZED SDK from https://www.stereolabs.com/developers/release
    2. pip install pyzed  (or run the get_python_api.py script from ZED SDK)

Usage:
    python test_zed.py
    python test_zed.py --save          # save all outputs to disk
    python test_zed.py --pointcloud    # also generate and display point cloud
"""

import sys
import argparse
import numpy as np
import cv2
import os

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed not installed.")
    print("Install ZED SDK from https://www.stereolabs.com/developers/release")
    print("Then run: python /usr/local/zed/get_python_api.py")
    sys.exit(1)


def test_camera_info(zed):
    """Print camera information and intrinsics."""
    info = zed.get_camera_information()
    params = info.camera_configuration.calibration_parameters

    print("\n" + "=" * 50)
    print("CAMERA INFO")
    print("=" * 50)
    print(f"Model:       {info.camera_model}")
    print(f"Serial:      {info.serial_number}")
    print(f"Firmware:    {info.camera_configuration.firmware_version}")
    print(f"Resolution:  {info.camera_configuration.resolution.width}x{info.camera_configuration.resolution.height}")
    print(f"FPS:         {info.camera_configuration.fps}")

    # Left camera intrinsics
    left = params.left_cam
    print(f"\nLeft Camera Intrinsics:")
    print(f"  Focal length: fx={left.fx:.1f}, fy={left.fy:.1f} pixels")
    print(f"  Principal pt: cx={left.cx:.1f}, cy={left.cy:.1f} pixels")
    print(f"  Distortion:   {left.disto}")

    # Right camera intrinsics
    right = params.right_cam
    print(f"\nRight Camera Intrinsics:")
    print(f"  Focal length: fx={right.fx:.1f}, fy={right.fy:.1f} pixels")
    print(f"  Principal pt: cx={right.cx:.1f}, cy={right.cy:.1f} pixels")

    # Stereo baseline
    baseline = params.get_camera_baseline()
    print(f"\nBaseline: {baseline:.2f} mm ({baseline/1000:.4f} m)")

    # Build K matrix (same format as Middlebury/FoundationStereo)
    K = np.array([
        [left.fx, 0, left.cx],
        [0, left.fy, left.cy],
        [0, 0, 1]
    ])
    print(f"\nIntrinsic matrix K:")
    print(K)

    return K, baseline


def test_capture(zed, save=False, out_dir="./zed_test_output"):
    """Capture and display color, depth, and stereo pair."""
    if save:
        os.makedirs(out_dir, exist_ok=True)

    # Allocate ZED Mat objects
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_map = sl.Mat()
    depth_display = sl.Mat()

    # Grab a frame
    runtime = sl.RuntimeParameters()
    err = zed.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Grab failed: {err}")
        return

    # ── Left RGB image ──
    zed.retrieve_image(left_image, sl.VIEW.LEFT)
    left_np = left_image.get_data()[:, :, :3].copy()  # BGRA → BGR
    print(f"\nLeft image:  {left_np.shape}, dtype={left_np.dtype}")

    # ── Right RGB image ──
    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
    right_np = right_image.get_data()[:, :, :3].copy()
    print(f"Right image: {right_np.shape}, dtype={right_np.dtype}")

    # ── Depth map (metric, in meters) ──
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
    depth_np = depth_map.get_data().copy()
    valid_depth = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
    print(f"Depth map:   {depth_np.shape}, dtype={depth_np.dtype}")
    print(f"  Valid pixels: {len(valid_depth)} / {depth_np.size}")
    if len(valid_depth) > 0:
        print(f"  Range: {valid_depth.min():.3f}m to {valid_depth.max():.3f}m")
        print(f"  Mean:  {valid_depth.mean():.3f}m")

    # ── Depth visualization ──
    zed.retrieve_image(depth_display, sl.VIEW.DEPTH)
    depth_viz = depth_display.get_data()[:, :, :3].copy()

    # ── Center pixel depth ──
    h, w = depth_np.shape[:2]
    center_depth = depth_np[h // 2, w // 2]
    print(f"\nCenter pixel depth: {center_depth:.3f}m")

    # Display
    combined_stereo = np.hstack([left_np, right_np])
    combined_stereo_small = cv2.resize(combined_stereo, (0, 0), fx=0.5, fy=0.5)

    combined_depth = np.hstack([left_np, depth_viz])
    combined_depth_small = cv2.resize(combined_depth, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Stereo Pair (Left | Right)", combined_stereo_small)
    cv2.imshow("Color | Depth", combined_depth_small)

    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if save:
        cv2.imwrite(os.path.join(out_dir, "left.png"), left_np)
        cv2.imwrite(os.path.join(out_dir, "right.png"), right_np)
        cv2.imwrite(os.path.join(out_dir, "depth_viz.png"), depth_viz)
        np.save(os.path.join(out_dir, "depth_meter.npy"), depth_np)

        # Save stereo pair side by side
        cv2.imwrite(os.path.join(out_dir, "stereo_pair.png"), combined_stereo)

        print(f"\nSaved outputs to {out_dir}/")

    return left_np, right_np, depth_np


def test_pointcloud(zed, save=False, out_dir="./zed_test_output"):
    """Generate and optionally display a point cloud."""
    import open3d as o3d

    point_cloud = sl.Mat()
    left_image = sl.Mat()

    runtime = sl.RuntimeParameters()
    err = zed.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Grab failed: {err}")
        return

    # Get point cloud (XYZRGBA format)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    zed.retrieve_image(left_image, sl.VIEW.LEFT)

    pc_data = point_cloud.get_data().copy()
    print(f"\nPoint cloud shape: {pc_data.shape}")

    # pc_data is (H, W, 4) where channels are X, Y, Z, RGBA_as_float
    xyz = pc_data[:, :, :3].reshape(-1, 3)

    # Extract colors from left image
    left_np = left_image.get_data()[:, :, :3].copy()
    colors = cv2.cvtColor(left_np, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    # Filter valid points
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.1) & (xyz[:, 2] < 5.0)
    xyz_valid = xyz[valid]
    colors_valid = colors[valid]
    print(f"Valid points: {xyz_valid.shape[0]} / {xyz.shape[0]}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)

    # Denoise
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    print(f"After denoising: {len(pcd.points)} points")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(out_dir, "cloud.ply"), pcd)
        print(f"Saved point cloud to {out_dir}/cloud.ply")

    # Visualize
    print("Showing point cloud. Press ESC to close.")
    o3d.visualization.draw_geometries([pcd])

    return pcd


def test_foundation_stereo_compatibility(zed, out_dir="./zed_test_output"):
    """
    Save stereo pair + intrinsics in the exact format FoundationStereo expects.
    You can then run FoundationStereo on these files.
    """
    os.makedirs(out_dir, exist_ok=True)

    info = zed.get_camera_information()
    params = info.camera_configuration.calibration_parameters
    left = params.left_cam
    baseline = params.get_camera_baseline() / 1000.0  # mm to meters

    # Build K matrix
    K = np.array([
        [left.fx, 0, left.cx],
        [0, left.fy, left.cy],
        [0, 0, 1]
    ])

    # Save K.txt for FoundationStereo
    k_path = os.path.join(out_dir, "K.txt")
    with open(k_path, "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
        f.write(f"{baseline:.6f}\n")

    # Capture and save stereo pair
    left_image = sl.Mat()
    right_image = sl.Mat()

    runtime = sl.RuntimeParameters()
    err = zed.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Grab failed: {err}")
        return

    zed.retrieve_image(left_image, sl.VIEW.LEFT)
    zed.retrieve_image(right_image, sl.VIEW.RIGHT)

    left_np = left_image.get_data()[:, :, :3].copy()
    right_np = right_image.get_data()[:, :, :3].copy()

    # ZED images are already rectified — FoundationStereo requires this
    cv2.imwrite(os.path.join(out_dir, "left.png"), left_np)
    cv2.imwrite(os.path.join(out_dir, "right.png"), right_np)

    print(f"\nFoundationStereo files saved to {out_dir}/")
    print(f"  left.png:  {left_np.shape}")
    print(f"  right.png: {right_np.shape}")
    print(f"  K.txt:     focal={left.fx:.1f}, baseline={baseline:.4f}m")
    print(f"\nRun FoundationStereo with:")
    print(f"  python scripts/run_demo.py \\")
    print(f"    --left_file {os.path.abspath(out_dir)}/left.png \\")
    print(f"    --right_file {os.path.abspath(out_dir)}/right.png \\")
    print(f"    --intrinsic_file {os.path.abspath(out_dir)}/K.txt \\")
    print(f"    --out_dir {os.path.abspath(out_dir)} \\")
    print(f"    --scale 0.5 --get_pc 1 --denoise_cloud 1")


def main():
    parser = argparse.ArgumentParser(description="Test ZED camera")
    parser.add_argument("--save", action="store_true", help="Save outputs to disk")
    parser.add_argument("--pointcloud", action="store_true", help="Generate and show point cloud")
    parser.add_argument("--foundation_stereo", action="store_true",
                        help="Save files for FoundationStereo")
    parser.add_argument("--out_dir", type=str, default="./zed_test_output")
    parser.add_argument("--resolution", type=str, default="HD720",
                        choices=["HD2K", "HD1080", "HD720", "VGA"],
                        help="Camera resolution")
    args = parser.parse_args()

    # ── Initialize ZED ──
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = True
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # best quality depth
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Set resolution
    res_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }
    init_params.camera_resolution = res_map[args.resolution]

    print(f"Opening ZED camera at {args.resolution}...")
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED: {err}")
        sys.exit(1)

    try:
        # Test 1: Camera info and intrinsics
        K, baseline = test_camera_info(zed)

        # Test 2: Capture color, depth, stereo
        test_capture(zed, save=args.save, out_dir=args.out_dir)

        # Test 3: Point cloud
        if args.pointcloud:
            test_pointcloud(zed, save=args.save, out_dir=args.out_dir)

        # Test 4: FoundationStereo compatibility
        if args.foundation_stereo:
            test_foundation_stereo_compatibility(zed, out_dir=args.out_dir)

    finally:
        zed.close()
        print("\nZED camera closed.")


if __name__ == "__main__":
    main()