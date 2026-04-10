import rtde_control
import rtde_receive
import time

# Connect to the robot
ROBOT_IP = "192.168.56.101"
# Creating interfaced for communication over TCP
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)


# ---- Read current state --------
# returns a list of 6 floats -> current angle of each joint in radians (base, shoulder, elbow, wrist1, wrist2, wrist3)
joint_positions = rtde_r.getActualQ()
# tool center point pose as 6 floats: x,y,z in meters relative to robot base, rx, ry, rz orientation as a rotation vector 
tcp_pose = rtde_r.getActualTCPPose()
print(f"Current joint poisitions (rad): {joint_positions}")
print(f"Current TCP pose [x,y,z,rx,ry,rz]: {tcp_pose}")

# ----- Move in joint space ------------
# Move to a "home" position (all joints in radians)
# -1.5708 < -90 in radians 
home = [0, -1.5708, 0, -1.5708, 0, 0]
velocity = 0.5 #rad/s
acceleration = 0.5 # rad/s^2
# Plans motion in Joint space
rtde_c.moveJ(home, velocity, acceleration)
print("Moved to home position")


# ----- Move in Cartesian space ------
# moveL target: [x, y, z, rx, ry, rz] in meters / radians
# Coordinates from base
# function blocks until target reached 
target_pose = [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04]
speed = 0.25 #m/s
accel = 1.2 # m/s^2
# Plans motion in Cartesian target
rtde_c.moveL(target_pose, speed, accel)
print("Moved to target pose")


# # --- Read force / torque ----
# # Returns a list of 6 floats [Fx, Fy, Fz, Tx, Ty, Tz] first three Newton forces along x,y,z . Torque in Nm around x,y,z
# wrench = rtde_r.getActualTCPForce()
# print(f"TCP force/torque: {wrench}")

# # --- Simple waypoint sequence ---
# # list of 4 cartesian poses. 
# waypoints = [
#     [-0.143, -0.435, 0.30, -0.001, 3.12, 0.04],
#     [-0.243, -0.435, 0.30, -0.001, 3.12, 0.04],
#     [-0.243, -0.535, 0.30, -0.001, 3.12, 0.04],
#     [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04],  # back to start
# ]

# # Iterates through the list. 
# for i, wp in enumerate(waypoints):
#     rtde_c.moveL(wp, 0.15, 0.8)
#     print(f"Reached waypoint {i+1}")
#     time.sleep(0.5)

# # --- Servo (real-time) control example ---
# # Useful for closed-loop / sensor-guided motion
# # grab current pose as starting point
# current = rtde_r.getActualTCPPose()
# dt = 0.002  # 500 Hz control loop ( matches UR controllers internal rate)
# lookahead = 0.1 # how much controller smooths incoming targets. (lower -> more responsive & jitterey)
# gain = 300 # how aggressive robot tracks the target, -> higher stiffer.

# for i in range(500):
#     t_start = time.time()
#     current[2] += 0.0001  # move 0.1 mm upward each step
#     rtde_c.servoL(current, 0, 0, dt, lookahead, gain)
#     elapsed = time.time() - t_start
#     if elapsed < dt:
#         time.sleep(dt - elapsed)

# rtde_c.servoStop()
# print("Servo motion complete")

# --- Cleanup ---
rtde_c.stopScript()
rtde_c.disconnect()
rtde_r.disconnect()
print("Disconnected")