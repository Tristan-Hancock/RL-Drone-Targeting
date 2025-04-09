# train_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from DroneNavEnv import DroneNavEnv

def main():
    # Create and wrap the environment.
    env = DroneNavEnv()
    monitored_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored_env])
    
    # Create PPO model with MLP policy.
    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, gamma=0.99, verbose=1, tensorboard_log="./tb_logs/airsim_drone/")
    
    # Train the model.
    total_timesteps = 100000  # Adjust this value as needed.
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    
    # Save the trained model.
    model.save("ppo_drone_nav")
    print("Training complete. Model saved as 'ppo_drone_nav.zip'.")

if __name__ == "__main__":
    main()
