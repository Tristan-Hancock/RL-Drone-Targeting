import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MinimalDroneEnv(gym.Env):
    """
    A minimal 3D drone environment without obstacles.
    The observation is the drone's 3D position. The action is a 3D continuous command.
    The reward is based on the negative distance to a goal location, and a large bonus is given
    when the goal is reached.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_steps = 300
        self.current_step = 0
        
        # Reward parameters.
        self.distance_multiplier = -1.0  # Strong negative reward for distance.
        self.success_bonus = 300.0       # Bonus when goal is reached.
        
        # Environment boundaries.
        self.boundary_min = np.array([-50, -50, 0])
        self.boundary_max = np.array([50, 50, 30])
        
        # Action space: 3D continuous.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Drone starting and goal positions.
        self.drone_start = np.array([-40.0, -40.0, 15.0])
        self.goal_position = np.array([40.0, 40.0, 2.0])
        self.goal_threshold = 3.0
        
        # We allow a relatively higher velocity for testing.
        self.max_velocity = 2.0
        
        # Observation is the drone’s position.
        self.observation_space = spaces.Box(low=self.boundary_min, high=self.boundary_max, dtype=np.float32)
        
        self.state = self.drone_start.copy()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = self.drone_start.copy()
        return self.state.copy(), {}
    
    def step(self, action):
        self.current_step += 1
        # Clamp the action to allowed range.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update state directly: new state = old state + (action * max_velocity * dt)
        self.state = self.state + action * self.max_velocity * self.dt
        # Clamp state within boundaries.
        self.state = np.clip(self.state, self.boundary_min, self.boundary_max)
        
        # Calculate Euclidean distance to goal.
        distance = np.linalg.norm(self.state - self.goal_position)
        reward = self.distance_multiplier * distance
        
        terminated = bool(distance < self.goal_threshold)
        if terminated:
            reward += self.success_bonus
        
        truncated = bool(self.current_step >= self.max_steps)
        info = {"step": self.current_step, "distance": distance}
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        pass

# Minimal test using a fixed (optimal) action.
if __name__ == '__main__':
    env = MinimalDroneEnv()
    obs, _ = env.reset()
    print("Initial Observation:", obs)
    
    done = False
    total_reward = 0.0
    
    while not done:
        # Compute the normalized direction toward the goal.
        direction = env.goal_position - obs
        norm = np.linalg.norm(direction)
        action = direction / norm if norm > 1e-6 else np.zeros(3)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {info['step']}: Obs = {obs}, Reward = {reward:.2f}, Distance = {info['distance']:.2f}")
        done = terminated or truncated

    print("Final total reward:", total_reward)
