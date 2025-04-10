import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim
import time

class DroneNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DroneNavEnv, self).__init__()
        # Observation: [dx, dy, dz, vx, vy, vz, yaw]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # Discrete Action Space: 7 actions (0: forward, 1: right, 2: backward, 3: left, 4: up, 5: down, 6: hover)
        self.action_space = spaces.Discrete(7)

        # Connect to AirSim.
        self.client = airsim.MultirotorClient()
        print("Connected to AirSim!")
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        # Episode parameters.
        self.max_steps = 500
        self.step_count = 0
        self.prev_distance = None

        # Fixed starting position (in NED, in meters) – adjust as needed.
        self.fixed_start = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        # Retrieve goal from object; if not found, use default.
        self.fixed_goal = self._get_goal_from_object()
        print(f"Goal position: {self.fixed_goal}")

        # Step parameters.
        self.step_length = 5.0  # m/s
        self.step_duration = 1.0  # seconds

    def _get_goal_from_object(self):
        """
        Try to retrieve the goal's pose from an object named 'GoalMarker'.
        If retrieval fails, use a default fixed goal.
        """
        try:
            pose = self.client.simGetObjectPose("GoalMarker")
            pos = pose.position
            if np.any(np.isnan([pos.x_val, pos.y_val, pos.z_val])):
                raise ValueError("Invalid goal pose")
            # Convert to Python floats.
            goal = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)], dtype=np.float32)
            return goal
        except Exception as e:
            print("Could not retrieve goal from object 'GoalMarker'; using default fixed goal. Error:", e)
            return np.array([121.0, -62.4, -14.48], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if not hasattr(self, 'episode_count'):
            self.episode_count = 0
        self.episode_count += 1
        print(f"[RESET] Episode {self.episode_count} starting...")
        print("[RESET] Resetting simulation...")
        self.client.reset()
        time.sleep(3)  # Allow simulator to stabilize after reset
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")
        self.step_count = 0

        # Set fixed start and goal positions.
        self.start_pos = self.fixed_start.copy()
        self.goal_pos = self.fixed_goal.copy()
        print(f"[RESET] Start pos: {self.start_pos}, Goal pos: {self.goal_pos}")

        # Set the drone's pose to the start position.
        start_pose = airsim.Pose(
            airsim.Vector3r(float(self.start_pos[0]), float(self.start_pos[1]), float(self.start_pos[2])),
            airsim.to_quaternion(0.0, 0.0, 0.0)
        )
        self.client.simSetVehiclePose(start_pose, True)
        time.sleep(1)
        self.client.hoverAsync(vehicle_name="Drone1").join()

        # Takeoff and ascend.
        print("[RESET] Taking off...")
        self.client.takeoffAsync(vehicle_name="Drone1").join()
        time.sleep(3)
        target_alt = float(self.start_pos[2])  # Should be -10.0
        print("[RESET] Ascending to start altitude...")
        self.client.moveToZAsync(target_alt, 3, vehicle_name="Drone1").join()
        time.sleep(3)
        self.client.hoverAsync(vehicle_name="Drone1").join()
        time.sleep(2)

        self.prev_distance = self._get_distance()
        return self._get_obs(), {}

    def _get_obs(self):
        """Fetch the drone's state and compute the observation vector."""
        state = self.client.getMultirotorState(vehicle_name="Drone1")
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
        drone_pos = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)], dtype=np.float32)
        rel_pos = self.goal_pos - drone_pos
        velocities = np.array([float(vel.x_val), float(vel.y_val), float(vel.z_val)], dtype=np.float32)
        obs = np.concatenate([rel_pos, velocities, np.array([float(yaw)], dtype=np.float32)])
        return obs

    def _get_distance(self):
        """Compute Euclidean distance from the drone to the goal."""
        obs = self._get_obs()
        return np.linalg.norm(obs[:3])

    def step(self, action):
        """
        Execute one discrete action step.
        Action mapping:
          0: Forward (North)
          1: Right (East)
          2: Backward (South)
          3: Left (West)
          4: Up (Ascend)
          5: Down (Descend)
          6: Hover
        """
        vx = vy = vz = 0.0
        if action == 0:   # Forward: +X in NED
            vx = self.step_length
        elif action == 1: # Right: +Y
            vy = self.step_length
        elif action == 2: # Backward: -X
            vx = -self.step_length
        elif action == 3: # Left: -Y
            vy = -self.step_length
        elif action == 4: # Up: Ascend (more negative Z in NED)
            vz = -self.step_length
        elif action == 5: # Down: Descend (increase Z)
            vz = self.step_length
        elif action == 6: # Hover
            vx = vy = vz = 0.0

        # Apply velocity command.
        self.client.moveByVelocityAsync(vx, vy, vz, duration=self.step_duration, vehicle_name="Drone1").join()
        self.step_count += 1

        obs = self._get_obs()
        current_distance = self._get_distance()
        reward = 0.0
        if self.prev_distance is not None:
            reward = (self.prev_distance - current_distance) * 10.0 - 1.0
        self.prev_distance = current_distance

        terminated = False
        if current_distance < 2.0:
            reward += 100.0
            terminated = True

        collision = self.client.simGetCollisionInfo(vehicle_name="Drone1")
        if collision.has_collided:
            reward -= 100.0
            terminated = True

        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        # Optional: add rendering code if desired.
        pass

    def close(self):
        print("[CLOSE] Disarming and resetting simulation...")
        self.client.armDisarm(False, vehicle_name="Drone1")
        self.client.enableApiControl(False, vehicle_name="Drone1")
        self.client.reset()


if __name__ == "__main__":
    env = DroneNavEnv()
    obs, _ = env.reset()
    print("Initial observation:", obs)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        print("Action:", action, "Obs:", obs, "Reward:", reward, "Done:", done)
        if done or truncated:
            break
    env.close()
