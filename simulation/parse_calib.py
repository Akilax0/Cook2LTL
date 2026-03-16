# Read each Middlebury scene calib.txt for required convertion

#Intrinsic file needs flattened 1x9 matrix 

import numpy as np 
import os

def parse_middlebury_calib(calib_path):
    params = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line: 
                key,val = line.split('=')
                key = key.strip()
                val = val.strip().strip('[]')
                # print("key,val",key,val)
                if key in ['cam0', 'cam1']:
                    # Parse the 3x3 matrix flattened with semicolons 
                    # (Intrinsic matrix)
                    rows = val.split(';')
                    # print("rows",rows)
                    matrix = []
                    for row in rows:
                        matrix.extend([float(x) for x in row.split()])
                    params[key] = matrix # 9 values

                else: 
                    try: 
                        params[key] = float(val)
                    except:
                        params[key] = val
        # print("line",line)
    return params

def create_foundation_stereo_intrinsics(calib_path, out_path):
    params = parse_middlebury_calib(calib_path)

    # cam0 is the left camera intrinsic matrix (flattened 1x9)
    cam0 = params['cam0'] # [fx, 0, cx, 0, fy, cy, 0, 0, 1] 
    
    #baseline: distance between the center points of two lenses or camera sensor in stereo system. 
    # Convertion because Middlebury gives it in mm -> m (foundation stereo)
    baseline_m = params['baseline']/1000.0

    with open(out_path, 'w') as f:
        f.write(' '.join(map(str, cam0)) + '\n')
        f.write(str(baseline_m) +'\n')
scene = "Adirondack-perfect"
print(scene)
scene_dir = os.path.abspath("./dataset/Middlebury/"+scene)

create_foundation_stereo_intrinsics(
    calib_path = f"{scene_dir}/calib.txt",
    out_path = f"{scene_dir}/intrinsics.txt"
)

