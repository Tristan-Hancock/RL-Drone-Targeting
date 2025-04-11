import time
import numpy as np
from stable_baselines3 import PPO
from Drone3DEnv import Drone3DEnv

def test_agent(model_path="./models/ppo_drone3d.zip", episodes=5):
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create the environment with rendering enabled (using human mode) and 10 buildings.
    env = Drone3DEnv(render_mode="human", num_buildings=10)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        print(f"Starting episode {episode + 1}...")
        while not done:
            # Predict action using the trained model (deterministic for evaluation).
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            print(f"Step {info['step']} - Reward: {reward:.2f}, Distance: {info['distance']:.2f}, Collision: {info['collision']}")
            done = terminated or truncated
            # Pause briefly to visually appreciate the simulation.
            time.sleep(0.05)
            
        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.2f}\n")
    
    env.close()

if __name__ == "__main__":
    test_agent()
