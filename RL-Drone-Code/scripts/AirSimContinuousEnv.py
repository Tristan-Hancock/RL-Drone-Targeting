import gym
from gym import spaces
import airsim
import numpy as np
import time

class AirSimContinuousEnv(gym.Env):
    """
    A custom Gym environment for continuous control of a drone using AirSim.
    """
    def __init__(self):
        super(AirSimContinuousEnv, self).__init__()
        print("[INIT] Initializing AirSimContinuousEnv...")

        # AirSim client setup
        self.client = airsim.MultirotorClient()
        print("[INIT] Confirming connection...")
        self.client.confirmConnection()
        print("[INIT] Enabling API control...")
        self.client.enableApiControl(True)
        print("[INIT] Arming drone...")
        self.client.armDisarm(True)

        # Continuous action space: vx, vy, vz (each in [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        print(f"[INIT] Action space set to: {self.action_space}")

        # Observation space: position (x, y, z) and velocity (vx, vy, vz)
        obs_low = np.array([-100, -100, -100, -10, -10, -10], dtype=np.float32)
        obs_high = np.array([100, 100, 100, 10, 10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        print(f"[INIT] Observation space set to: {self.observation_space}")

        # Set the velocity scaling factor (adjust as needed)
        self.max_velocity = 20  # e.g., 20 m/s for testing
        print(f"[INIT] max_velocity set to: {self.max_velocity}")

        # Set a simplified target position (closer for easier learning)
        self.target_position = np.array([5, 0, -10])  # target near the starting altitude
        print(f"[INIT] target_position set to: {self.target_position}")

        # For incremental reward shaping, store the previous distance.
        self.prev_distance = None

    def reset(self):
        print("[RESET] Resetting simulation...")
        # Reset the simulation to a fixed starting point.
        self.client.reset()
        time.sleep(0.5)  # Allow some time for the simulation to reset

        print("[RESET] Taking off...")
        # Take off
        self.client.takeoffAsync().join()
        print("[RESET] Drone took off.")

        # Move to a higher altitude using an absolute position command.
        print("[RESET] Moving to altitude -10...")
        self.client.moveToPositionAsync(0, 0, -10, 5).join()
        print("[RESET] Reached altitude -10.")

        # Initialize previous distance based on current observation
        obs = self._get_obs()
        pos = np.array([obs[0], obs[1], obs[2]])
        self.prev_distance = np.linalg.norm(pos - self.target_position)
        print(f"[RESET] Initial position: {obs[:3]}, Initial distance to target: {self.prev_distance}")
        return obs

    def _get_obs(self):
        # Get the drone's state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        observation = np.array([pos.x_val, pos.y_val, pos.z_val,
                                vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        return observation

    def step(self, action):
        print(f"[STEP] Received action: {action}")
        # Convert action to native Python floats and scale them
        vx = float(self.max_velocity * action[0])
        vy = float(self.max_velocity * action[1])
        vz = float(self.max_velocity * action[2])
        print(f"[STEP] Scaled velocities: vx={vx}, vy={vy}, vz={vz}")

        # Use body-frame velocity command for intuitive control
        print("[STEP] Sending velocity command...")
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration=3).join()
        time.sleep(1)  # Allow some time for the movement to register

        # Get new observation
        obs = self._get_obs()
        pos = np.array([obs[0], obs[1], obs[2]])
        current_distance = np.linalg.norm(pos - self.target_position)
        print(f"[STEP] New position: {obs[:3]}, Target distance: {current_distance}")

        # Incremental reward: reward for reducing the distance, minus a time penalty
        reward = (self.prev_distance - current_distance) * 10 - 0.1
        print(f"[STEP] Reward calculated: {reward}")
        self.prev_distance = current_distance

        # Episode is done if the drone is within 1.0 m of the target.
        done = current_distance < 1.0
        if done:
            print("[STEP] Episode complete: target reached.")

        info = {"distance": current_distance}
        return obs, reward, done, info

    def render(self, mode='human'):
        # Rendering can be implemented if needed (e.g., retrieving camera images)
        print("[RENDER] Render not implemented.")

    def close(self):
        print("[CLOSE] Disarming and disabling API control...")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        print("[CLOSE] Environment closed.")

# Testing the environment directly (for debugging)
if __name__ == "__main__":
    env = AirSimContinuousEnv()
    obs = env.reset()
    print("Initial observation:", obs[:3])
    for _ in range(10):
        action = env.action_space.sample()  # random action for testing
        print("Action sampled:", action)
        obs, reward, done, info = env.step(action)
        print("New observation:", obs[:3], "Reward:", reward, "Done:", done, "Info:", info)
        if done:
            print("[TEST] Done condition met. Breaking out of loop.")
            break
    env.close()
