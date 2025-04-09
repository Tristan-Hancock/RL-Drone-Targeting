# DroneNavEnv.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim
import time

class DroneNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DroneNavEnv, self).__init__()
        # Observation: [dx, dy, dz, vx, vy, vz, yaw, front_distance]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # Action: [pitch, roll, yaw_rate, throttle] each normalized in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        print("Connected!")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Episode settings
        self.max_steps = 300
        self.step_count = 0

    def reset(self, seed=None, options=None):
        print("[RESET] Resetting simulation...")
        self.client.reset()
        # Extra delay to allow simulator to stabilize
        time.sleep(3)
        self.step_count = 0

        # Define safe altitude (NED: negative means above ground)
        safe_altitude = -10.0
        
        # Randomize start and goal positions (x, y between -5 and 5) with fixed safe altitude.
        self.start_pos = np.array([np.random.uniform(-5, 5), 
                                    np.random.uniform(-5, 5), 
                                    safe_altitude])
        self.goal_pos  = np.array([np.random.uniform(-5, 5), 
                                    np.random.uniform(-5, 5), 
                                    safe_altitude])
        print(f"[RESET] Start pos: {self.start_pos}, Goal pos: {self.goal_pos}")

        # Set vehicle pose to the start position.
        start_pose = airsim.Pose(airsim.Vector3r(*self.start_pos), airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(start_pose, True)
        time.sleep(1)
        self.client.hoverAsync().join()

        # Attempt takeoff with error handling.
        try:
            print("[RESET] Taking off...")
            self.client.takeoffAsync().join(timeout=10)
        except Exception as e:
            print("TakeoffAsync timed out or failed:", e)

        try:
            print("[RESET] Moving to safe altitude...")
            self.client.moveToZAsync(safe_altitude, 2).join(timeout=10)
        except Exception as e:
            print("moveToZAsync timed out or failed:", e)
            
        time.sleep(1)  # Additional delay for stabilization

        # Get initial state and initialize previous distance for reward shaping.
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
        throttle_cmd = 0.5 + 0.5 * float(action[3])  # scale throttle between 0 and 1

        # Use the API to move the drone.
        self.client.moveByRollPitchYawrateThrottleAsync(
            roll_cmd,
            pitch_cmd,
            yaw_rate_cmd,
            throttle_cmd,
            0.1
        ).join()
        self.step_count += 1

        # Get updated state.
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

        # Compute reward based on progress toward the goal.
        current_dist = np.linalg.norm(np.array([
            self.goal_pos[0] - pos.x_val,
            self.goal_pos[1] - pos.y_val,
            self.goal_pos[2] - pos.z_val
        ]))
        reward = self.prev_dist_to_goal - current_dist  # Positive if closer.
        self.prev_dist_to_goal = current_dist

        # Check termination conditions.
        terminated = False
        truncated = False
        if current_dist < 1.0:
            reward += 100.0  # Goal reached bonus.
            terminated = True

        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 100.0  # Collision penalty.
            terminated = True

        # Small time penalty.
        reward -= 0.01

        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def _get_front_distance(self):
        """Placeholder for a front sensor reading. Replace with actual sensor data if available."""
        return 10.0

    def render(self, mode='human'):
        # Optional: implement camera capture if desired.
        pass

    def close(self):
        print("[CLOSE] Disarming and resetting simulation...")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()

# Quick test:
if __name__ == "__main__":
    env = DroneNavEnv()
    obs, _ = env.reset()
    print("Initial observation:", obs)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step reward:", reward, "Terminated:", terminated, "Truncated:", truncated)
        if terminated or truncated:
            break
    env.close()
