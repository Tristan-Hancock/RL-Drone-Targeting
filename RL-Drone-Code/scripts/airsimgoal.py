import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim
import time

class AirSimDroneEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()
        # Observation: [dx, dy, dz, vx, vy, vz, yaw, front_distance]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # Action: [pitch, roll, yaw_rate, throttle] each normalized in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        print("Connected to AirSim!")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Episode settings
        self.max_steps = 300
        self.step_count = 0

        # Add episode count attribute.
        self.episode_count = 0

        # Fixed start and goal positions (you can update these to your desired coordinates)
        self.fixed_start = np.array([0.0, 0.0, -20.0])
        self.fixed_goal  = np.array([62.4, -121.0, -14.48])

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        print(f"[RESET] Starting episode {self.episode_count}")
        print("[RESET] Resetting simulation...")
        self.client.reset()
        time.sleep(3)  # Allow simulator to stabilize
        self.step_count = 0

        # Use fixed start and goal positions.
        self.start_pos = self.fixed_start.copy()
        self.goal_pos = self.fixed_goal.copy()
        print(f"[RESET] Start pos: {self.start_pos}, Goal pos: {self.goal_pos}")

        # Set the drone's pose to the start position.
        start_pose = airsim.Pose(airsim.Vector3r(*self.start_pos),
                                  airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(start_pose, True)
        time.sleep(1)
        self.client.hoverAsync().join()

        # Execute takeoff and ascend to safe altitude.
        print("[RESET] Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        print("[RESET] Ascending to start altitude...")
        self.client.moveToZAsync(self.start_pos[2], 2).join()
        time.sleep(2)
        self.client.hoverAsync().join()
        time.sleep(2)

        # Get initial state.
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        front_dist = self._get_front_distance()
        self.prev_dist_to_goal = np.linalg.norm(self.goal_pos - np.array([pos.x_val, pos.y_val, pos.z_val]))
        obs = np.array([
            self.goal_pos[0] - pos.x_val,
            self.goal_pos[1] - pos.y_val,
            self.goal_pos[2] - pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            yaw,
            front_dist
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Unpack and scale the action command:
        pitch_cmd    = float(action[0]) * 0.2618   # ~15 degrees in radians
        roll_cmd     = float(action[1]) * 0.2618
        yaw_rate_cmd = float(action[2]) * 45.0       # degrees per second
        throttle_cmd = 0.5 + 0.5 * float(action[3])  # scale throttle to [0, 1]

        # Apply control command.
        self.client.moveByRollPitchYawrateThrottleAsync(
            roll_cmd,
            pitch_cmd,
            yaw_rate_cmd,
            throttle_cmd,
            0.1
        ).join()
        self.step_count += 1

        # Retrieve updated state.
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        front_dist = self._get_front_distance()
        obs = np.array([
            self.goal_pos[0] - pos.x_val,
            self.goal_pos[1] - pos.y_val,
            self.goal_pos[2] - pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            yaw,
            front_dist
        ], dtype=np.float32)

        # Reward: measure change in distance to the goal.
        current_dist = np.linalg.norm(np.array([
            self.goal_pos[0] - pos.x_val,
            self.goal_pos[1] - pos.y_val,
            self.goal_pos[2] - pos.z_val
        ]))
        reward = self.prev_dist_to_goal - current_dist
        self.prev_dist_to_goal = current_dist

        terminated = False
        truncated = False
        if current_dist < 1.0:
            reward += 100.0  # bonus for reaching the goal
            terminated = True

        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 100.0  # penalty for collision
            terminated = True

        reward -= 0.01  # small time penalty

        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def _get_front_distance(self):
        """Placeholder for a front sensor reading."""
        return 10.0

    def render(self, mode='human'):
        pass

    def close(self):
        print("[CLOSE] Disarming and resetting simulation...")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()

# Quick test script:
if __name__ == "__main__":
    env = AirSimDroneEnv()
    obs, _ = env.reset()
    print("Initial observation:", obs)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print("Step reward:", reward, "Terminated:", terminated, "Truncated:", truncated)
        if terminated or truncated:
            break
    env.close()
