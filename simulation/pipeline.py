import os,subprocess
import re
import logging
import torch
from omegaconf import OmegaConf
from ultralytics import YOLO
import cv2
import numpy as np
import open3d as o3d

# #To import the modules
# FOUNDATION_DIR = os.path.abspath("./FoundationStereo")

# if FOUNDATION_DIR not in sys.path:
#     sys.path.append(FOUNDATION_DIR)

# from core.foundation_stereo import FoundationStereo
# from core.utils.utils import InputPadder


def parse_calib(calib_path):
    """
    
    ================Parse Middlebury calib.txt ================

    cam0,1: camera matrices for the rectified views [f 0 cx; 0 f cy; 0 0 1]
        f: focal length in pixels
        cx, cy: principal point (differs in views 0 and 1)
    doffs: x-difference of principal points, doffs = cx1 - cx0
    baseline: camera baseline in mm
    w, h : img size
    ndsp: a bound for disparity levels (search from d=0...ndisp-1)
    isint: if GT disparities only have integer precision (older datasets)
    vmin, vmax: tight bound for min max disparities, for color visualization;
    dyavg, dymax: avaerage maximum absolute y disaprities, indicationof calivration error \
    in imperfect datasets.

    from floating point disparity d(pixels) .pfm to depth Z(mm)
        Z = baseline * f / (d +doffs)
    
    """
    calib = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=",1)
            key = key.strip()
            value = value.strip()
            if key in ("cam0", "cam1"):
                import re
                nums = list(map(float, re.findall(r"[-\d.]+", value)))
                calib[key] = np.array(nums).reshape(3,3)
            elif key in ("doffs", "baseline", "vmin", "vmax"):
                calib[key] = float(value)
            elif key in ("width", "height", "ndisp"):
                calib[key] = int(value)

    return calib


def make_intrinsic_file(calib, out_path):
    """
    Convert Middlebury calib.txt into Foundation Stereo's K.txt format.
    Line 1: flattened 1x9 intrisic matrix (from cam0)
    Line 2: baseline in meters
    """
    
    K = calib["cam0"]
    baseline_m = calib["baseline"] / 1000.0 # middlebury mm -> FS meters

    
    with open(out_path, "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
        f.write(f"{baseline_m:.6f}\n")

    return out_path


def visualize_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))

    # Apply rotation around the center of the point cloud
    # pcd.rotate(R, center=pcd.get_center())
    pcd.rotate(R)

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()



# def load_foundation_stereo(ckpt_path):
#     """
#     ====== Load FoundationStereo model from checkpoint =======

#     """

#     cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
#     cfg = OmegaConf.load(cfg_path)

#     if "vit_size" not in cfg:
#         cfg["vit_size"] = "vitl"


#     model = FoundationStereo(cfg)
#     ckpt = torch.load(ckpt_path)
#     model.load_state_dict(ckpt["model"])
#     model.cuda()
#     model.eval()
#     return model, cfg

# def run_foundation_stereo(model, left_img, right_img, scale=1.0, valid_iters=32):
#     """
#     Run Foundation on a stereo pair.

#     Args:
#         model: Loaded Foundation model
#         left_img: Left image as np array (H, W, 3) BGR
#         right_img: Right image as np array  (H, W, 3) BGR
#         scale: Downscale factor for faster inference
#         valid_iters: Number of refinement iterations

#     Returns: 
#         disparity: (H, W) numpy array of disparity values    
    
#     """

#     # BGR -> RGB , HWC -> CHW, normalizze to [0,1]
#     left  = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
#     right = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

#     left = torch.from_numpy(left).permute(2, 0, 1).float()/255.0
#     right = torch.from_numpy(right).permute(2, 0, 1).float()/255.0
    
#     # Optional downscale
#     if scale != 1.0:
#         _, h, w = left.shape
#         new_h, new_w = int(h*scale), int(w* scale)
#         left = torch.nn.functional.interpolate(
#             left.unsqueeze(0), size=(new_h, new_w),mode="bilinear", align_corners=True).squeeze(0)
#         right = torch.nn.functional.interpolate(
#             right.unsqueeze(0), size=(new_h, new_w),mode="bilinear", align_corners=True).squeeze(0)
        
#     # pad to be divisible by 32
#     left = left.unsqueeze(0).cuda()
#     right = right.unsqueeze(0).cuda()
#     padder = InputPadder(left.shape)
#     left, right = padder.pad(left, right)
    
#     with torch.no_grad():
#         disp = model.forward(left, right, iters=valid_iters)


#     #unpad
#     disp = padder.unpad(disp)
#     disp = disp.squeeze().cpu().numpy()


#     #Scale disparity back if we downscaled the image
#     if scale != 1.0:
#         disp = cv2.resize(disp, (left_img.shape[1], left_img.shape[0]))/scale

#     return disp

def crop_stereo_pair(left_path, right_path, box, out_dir,calib_path=None, padding=20):
    """
    Crop both stereo images at the same coordinates. 
    padding : extra pixels around the bounding box
    """

    os.makedirs(out_dir, exist_ok=True)

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    print("left shape:",left.shape)
    H,W = left.shape[:2]

    # parse ndisp (maximum disparity object can be shifted) from calib.txt
    # shifting based on depth 
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            for line in f:
                if line.startswith("ndisp"):
                    ndisp = int(line.split("=")[1])
                    break

    print(f"Using ndisp={ndisp} for right image offset")

    x1, y1, x2, y2 = box

    # left crop std padding
    lx1 = max(0, x1-padding)
    ly1 = max(0, y1-padding)
    lx2 = min(W, x2+padding)
    ly2 = min(H, y2+padding)
    
    #Extending right crop
    rx1 = max(0,x1-padding-ndisp)
    ry1 = ly1
    rx2 = min(W, x2+padding)
    ry2 = ly2

    print("left:",left.shape) 
    #Crop both IDENTICAL coordinates
    left_crop = left[ly1:ly2,lx1:lx2]
    right_crop = right[ry1:ry2, rx1:rx2]

    # Saving the crops
    left_out = os.path.join(out_dir, "crop_im0.png")
    right_out = os.path.join(out_dir, "crop_im1.png")
    cv2.imwrite(left_out, left_crop)
    cv2.imwrite(right_out, right_crop)

    return x1, y1, x2, y2, left_out, right_out

# Config

SCENE_DIR = os.path.abspath("../../dataset/Middlebury/Adirondack-perfect")
FOUNDATION_DIR = os.path.abspath("./FoundationStereo")
CKPT = f"{FOUNDATION_DIR}/pretrained_models/23-51-11/model_best_bp2-001.pth"
TARGET_CLASS = "cup"
OUT_DIR = "./object_depth_output"
os.makedirs(OUT_DIR, exist_ok =True)

SCALE = 0.25
valid_iters = str(16)

#YOLO detection
print("Running YOLO26n to detect")
model = YOLO("yolo26n.pt")
results = model(f"{SCENE_DIR}/im0.png", conf=0.5)#confidence threshold
# print(results[0].boxes)

#Finding the target object

target_box = None
for box in results[0].boxes:
    cls = results[0].names[int(box.cls)]
    # print(box)
    if TARGET_CLASS.lower() in cls.lower():
        target_box = list(map(int, box.xyxy[0].tolist()))
        print(f"Found: {cls} at {target_box}")
        break
    
if target_box is None:
    print(f"{TARGET_CLASS} not found. Available:")
    for box in results[0].boxes:
        print(f"  - {results[0].names[int(box.cls)]}")
    exit()


# Running FoundationStereo
# print("Loading Foundation Stereo")
# fs_model, fs_cfg = load_foundation_stereo(CKPT)

# left_img = cv2.imread(f"{SCENE_DIR}/im0.png")
# right_img = cv2.imread(f"{SCENE_DIR}/im1.png")

# print("Stereo matching on full images")
# disparity = run_foundation_stereo(fs_model, left_img, right_img, scale=0.5)
# print(f"Disparity map shape: {disparity.shape}, range: [{disparity.min():.1f},\
#       {disparity.max():.1f}]")

# # save disparity visualization 
# disp_viz  = (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255 
# cv2.imwrite(os.path.join(OUT_DIR, "disparity_full.png"), disp_viz.astype(np.uint8))

calib = parse_calib(f"{SCENE_DIR}/calib.txt")
intrinsic_file = os.path.join(OUT_DIR, "K.txt")
make_intrinsic_file(calib, intrinsic_file)
print(f"created intrinsic file: {intrinsic_file}")

print("calib : ", calib)

# Run FoundationStereo via subprocess

cmd = [
    "python3", f"{FOUNDATION_DIR}/scripts/run_demo.py",
    "--left_file", f"{SCENE_DIR}/im0.png",
    "--right_file", f"{SCENE_DIR}/im1.png",
    "--ckpt_dir", CKPT,
    "--out_dir", OUT_DIR,
    "--intrinsic_file", intrinsic_file,
    "--scale", str(SCALE),
    "--valid_iters", valid_iters
]

print(f"Running:{' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode !=0:
    print(f"ERROR:\n{result.stderr}")
    exit()

 
# visualize_point_cloud(f"{OUT_DIR}/cloud.ply")


# Load disparity + extract object point cloud
depth_path = os.path.join(OUT_DIR, "depth_meter.npy")
cloud_path = os.path.join(OUT_DIR, "cloud_denoise.ply")

#scale YOLO bbox to match the dowscaled depth/point cloud
x1, y1, x2, y2 = target_box
x1_s = int(x1*SCALE)
y1_s = int(y1*SCALE)
x2_s = int(x2*SCALE)
y2_s = int(y2*SCALE)


# Scaled intrinsics 
focal_s = calib["cam0"][0, 0] * SCALE
cx_s = calib["cam0"][0, 2] * SCALE
cy_s = calib["cam0"][1, 2] * SCALE

# Crop stereo pair 

x1, y1, x2, y2, lp, rp =crop_stereo_pair(
    left_path = f"{SCENE_DIR}/im0.png",
    right_path = f"{SCENE_DIR}/im1.png",
    box = target_box,
    out_dir = OUT_DIR,
    calib_path=f"{SCENE_DIR}/calib.txt"
)



