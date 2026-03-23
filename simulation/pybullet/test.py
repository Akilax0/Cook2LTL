import pybullet as p
import pybullet_data
import time


# Connect to PyBullet GUI
# opens a 3D viewer window 
client = p.connect(p.GUI)
# pybullet_data ships with common assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#Gravity towards Z
p.setGravity(0,0, -9.81)

# Loading ground plane 
plane_id = p.loadURDF("plane.urdf")

#Load UR10 using robot descriptions 
from robot_descriptions.loaders.pybullet import load_robot_description
robot_id = load_robot_description("ur10_description", useFixedBase=True)

#Inspecting Joints
num_joints = p.getNumJoints(robot_id)
print(f"\nUR10 loaded - {num_joints} joints total\n")
print(f"{'Index':<6} {'Name':<25} {'Type':<10} {'Lower':<10} {'Upper':<10} {'Parent Link':<20} {'Child Link':<20}")
print("-" * 101)

arm_joints = []
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_index = info[0]
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    lower_limit = info[8]
    upper_limit = info[9]
    parent_link = info[12].decode("utf-8")
    child_link = info[12].decode("utf-8")

    type_name = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(joint_type, str(joint_type))

    print(f"{joint_index:<6} {joint_name:<25} {type_name:<10} {lower_limit:<10.3f} {upper_limit:<10.3f} {parent_link:<20}")

    if joint_type == p.JOINT_REVOLUTE:
        arm_joints.append(joint_index)

print(f"\nArm joint indices: {arm_joints}")
print(f"End-effector link index: {arm_joints[-1] + 1 if arm_joints else 'N/A'}")
# ── Get initial end-effector pose ──
ee_index = arm_joints[-1]
ee_state = p.getLinkState(robot_id, ee_index)
print(f"\nEE position: {[f'{x:.3f}' for x in ee_state[4]]}")
print(f"EE orientation (quat): {[f'{x:.3f}' for x in ee_state[5]]}")



while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)