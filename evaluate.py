import time

import torch
import tqdm

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import (
    DynamixelMotorsBusConfig, TorqueMode)
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.scripts.control_robot import busy_wait

leader_port = "/dev/tty.usbmodem578E0214181"
follower_port = "/dev/tty.usbmodem58760435411"

leader_arm = DynamixelMotorsBusConfig(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBusConfig(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

robot = ManipulatorRobot(
    KochRobotConfig(
        leader_arms={"main": leader_arm},
        follower_arms={"main": follower_arm},
        calibration_dir=".cache/calibration/koch",
        cameras={
            "table": OpenCVCameraConfig(0, fps=30, width=640, height=480),
        },
    )
)

print("Connecting to robot...")
robot.connect()
print("Connected to robot.")

inference_time_s = 60
fps = 30
device = "mps"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = (
    "outputs/train/koch_stacking_same_lego_pi0/checkpoints/last/pretrained_model"
)
print(f"Loading policy from {ckpt_path}...")
policy = PI0Policy.from_pretrained(ckpt_path)
print("Policy loaded.")
policy.to(device)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    observation["task"] = ["stack the lego on top of the other one"]

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

robot.disconnect()
