import time
from stable_baselines3 import PPO
from Drone3DEnv import Drone3DEnv

def test_agent(model_path="./models/ppo_drone3d.zip", episodes=5):
    model = PPO.load(model_path)
    env = Drone3DEnv(render_mode="human")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        print(f"Starting episode {ep+1}...")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            print(f"Step {info['step']} - Reward: {reward:.2f}, Distance: {info['distance']:.2f}, Collision: {info['collision']}")
            done = terminated or truncated
            time.sleep(0.05)
        if terminated:
            print(f"Episode {ep+1}: Success! Reached base camp with total reward: {episode_reward:.2f}")
        else:
            print(f"Episode {ep+1}: Failed to reach base camp. Total reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test_agent()
