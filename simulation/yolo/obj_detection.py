from ultralytics import YOLO
import os
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")

# Get the name of the current GPU
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load the new YOLO26 model
model = YOLO("yolo26n.pt").to("cuda") 

print(f"Model is on: {model.device}")

# folder_path = "../dataset/KITchen_rgb_1/000/"
folder_path = "../dataset/Middlebury/images/left"
output_folder = "middlebury"


# Running batch inference
results = model.predict(folder_path,device=0,half=True, batch=256, save= True, project='test_runs', name=output_folder,exist_ok=True,stream=True) 

for r in results:
    pass

print("done")
# for i,r in enumerate(results):
#     print(f"Image {i}: {r.path} -> Dound {len(r.boxes)} objects")

# for filename in os.listdir(folder_path):
#     if filename.lower().endswith("png"):
#         img_path = os.path.join(folder_path, filename)
        
#         # Run inference
#         results = model.predict(img_path,device=0,half=True, batch=16) 

#         for i, r in enumerate(results):
#             r.save(filename=os.path.join(output_folder, f"det_{filename}"))
#         # Example: Print how many objects were found in THIS image
#         # for r in results:
#         #     print(f"{filename}: Found {len(r.boxes)} objects")
#         #     r.show()

