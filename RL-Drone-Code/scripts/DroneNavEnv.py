import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim
import time
import cv2  # Ensure opencv-python is installed

class DroneNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DroneNavEnv, self).__init__()
        # Observation: an RGB image from the front camera of shape (3, 50, 50)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 50, 50), dtype=np.uint8)
        # Discrete Action Space: 9 actions (0 to 8)
        self.action_space = spaces.Discrete(9)

        # Connect to AirSim.
        self.client = airsim.MultirotorClient()
        print("Connected to AirSim!")
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        # Episode parameters.
        self.max_steps = 500
        self.step_count = 0

        # Fixed starting position (in NED, in meters).
        self.fixed_start = np.array([0.0, 0.0, -50.0], dtype=np.float32)
        # Retrieve goal from object; if not found, use default.
        self.fixed_goal = self._get_goal_from_object()
        print(f"Goal position: {self.fixed_goal}")

        # Step parameters.
        self.step_length = 5.0  # m/s
        self.step_duration = 0.2  # seconds

    def _get_goal_from_object(self):
        try:
            pose = self.client.simGetObjectPose("GoalMarker")
            pos = pose.position
            if np.any(np.isnan([pos.x_val, pos.y_val, pos.z_val])):
                raise ValueError("Invalid goal pose")
            goal = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)], dtype=np.float32)
            return goal
        except Exception as e:
            print("Could not retrieve goal from object 'GoalMarker'; using default fixed goal. Error:", e)
            return np.array([121.0, -62.4, -14.48], dtype=np.float32)

    def reset(self, seed=None, options=None):
        try:
            if not hasattr(self, 'episode_count'):
                self.episode_count = 0
            self.episode_count += 1
            print(f"[RESET] Episode {self.episode_count} starting...")
            print("[RESET] Resetting simulation...")
            self.client.reset()
            time.sleep(3)
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            self.step_count = 0

            # Use new fixed starting position.
            new_start_pos = np.array([5.0, -2.0, -15.0], dtype=np.float32)
            self.start_pos = new_start_pos.copy()
            self.goal_pos = self.fixed_goal.copy()  # Goal remains as previously retrieved.
            print(f"[RESET] Start pos: {self.start_pos}, Goal pos: {self.goal_pos}")

            # Set the drone's pose to the new start position.
            start_pose = airsim.Pose(
                airsim.Vector3r(float(new_start_pos[0]), float(new_start_pos[1]), float(new_start_pos[2])),
                airsim.to_quaternion(0.0, 0.0, 0.0)
            )
            self.client.simSetVehiclePose(start_pose, True)
            time.sleep(1)
            self.client.hoverAsync(vehicle_name="Drone1").join()

            # Takeoff and ascend.
            print("[RESET] Taking off...")
            self.client.takeoffAsync(vehicle_name="Drone1").join()
            time.sleep(3)
            target_alt = float(new_start_pos[2])  # Using the new starting altitude (-15.0)
            print("[RESET] Ascending to start altitude...")
            self.client.moveToZAsync(target_alt, 3, vehicle_name="Drone1").join()
            time.sleep(3)
            self.client.hoverAsync(vehicle_name="Drone1").join()
            time.sleep(2)

            # Compute initial distance for reward calculation.
            self.prev_distance = self._get_distance()
            obs = self._get_obs()
            if obs is None:
                raise ValueError("Observation is None after reset.")
            return obs, {}
        except Exception as e:
            print("Error during reset:", e)
            default_obs = np.zeros((3, 50, 50), dtype=np.uint8)
            return default_obs, {}

    def _get_obs(self):
        """
        Retrieve an RGB image from the front camera,
        resize it to 50x50, and return it in channel-first format (3, 50, 50).
        """
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        if responses and responses[0].width > 0 and responses[0].height > 0:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            img_resized = cv2.resize(img_rgb, (50, 50), interpolation=cv2.INTER_AREA)
            img_transposed = np.transpose(img_resized, (2, 0, 1))
            return img_transposed
        else:
            return np.zeros((3, 50, 50), dtype=np.uint8)

    def _get_distance(self):
        state = self.client.getMultirotorState(vehicle_name="Drone1")
        pos = state.kinematics_estimated.position
        drone_pos = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)], dtype=np.float32)
        return np.linalg.norm(self.goal_pos - drone_pos)

    def step(self, action):
        """
        Execute one discrete action step.
        Action mapping (9 actions):
          0: Forward-Left (diagonal)
          1: Forward
          2: Forward-Right (diagonal)
          3: Left
          4: Hover (no movement)
          5: Right
          6: Backward-Left (diagonal)
          7: Backward
          8: Backward-Right (diagonal)
        """
        vx, vy, vz = 0.0, 0.0, 0.0
        speed = self.step_length

        if action == 0:  # Forward-Left
            vx = speed
            vy = -speed
        elif action == 1:  # Forward
            vx = speed
            vy = 0.0
        elif action == 2:  # Forward-Right
            vx = speed
            vy = speed
        elif action == 3:  # Left
            vx = 0.0
            vy = -speed
        elif action == 4:  # Hover
            vx = 0.0
            vy = 0.0
        elif action == 5:  # Right
            vx = 0.0
            vy = speed
        elif action == 6:  # Backward-Left
            vx = -speed
            vy = -speed
        elif action == 7:  # Backward
            vx = -speed
            vy = 0.0
        elif action == 8:  # Backward-Right
            vx = -speed
            vy = speed

        vz = 0.0

        self.client.moveByVelocityAsync(vx, vy, vz, duration=self.step_duration, vehicle_name="Drone1").join()
        self.step_count += 1

        obs = self._get_obs()
        current_distance = self._get_distance()
        distance_improvement = self.prev_distance - current_distance
        reward = distance_improvement * 10.0
        if distance_improvement < 0:
            reward -= 5.0
        if current_distance < 2.0:
            reward += 200.0

        collision = self.client.simGetCollisionInfo(vehicle_name="Drone1")
        terminated = collision.has_collided
        if terminated:
            reward -= 200.0

        self.prev_distance = current_distance
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        pass

    def close(self):
        print("[CLOSE] Disarming and resetting simulation...")
        self.client.armDisarm(False, vehicle_name="Drone1")
        self.client.enableApiControl(False, vehicle_name="Drone1")
        self.client.reset()

if __name__ == "__main__":
    env = DroneNavEnv()
    obs, _ = env.reset()
    print("Initial observation shape:", obs.shape)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        print("Action:", action, "Obs shape:", obs.shape, "Reward:", reward, "Done:", done)
        if done or truncated:
            break
    env.close()
