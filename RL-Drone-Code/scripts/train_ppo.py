# train_ppo.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from DroneNavEnv import DroneNavEnv
import os
import time

# Custom callback to print current timestep every 100 calls.
class PrintTimestepsCallback(BaseCallback):
    def __init__(self, print_freq=100, verbose=0):
        super(PrintTimestepsCallback, self).__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            print(f"Training timestep: {self.model.num_timesteps}")
        return True

def create_env():
    env = DroneNavEnv()
    monitored_env = Monitor(env, filename="./tb_logs/airsim_drone/monitor.csv")
    vec_env = DummyVecEnv([lambda: monitored_env])
    return vec_env

def main():
    # Create necessary directories.
    log_dir = "./tb_logs/airsim_drone/"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create the environment.
    vec_env = create_env()
    
    # Set up callbacks: Checkpoint saves every 1000 timesteps,
    # and PrintTimestepsCallback prints current timestep every 100 steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=10, 
        save_path=checkpoint_dir, 
        name_prefix="ppo_drone"
    )
    print_callback = PrintTimestepsCallback(print_freq=100, verbose=1)
    
    # Create PPO model with the MlpPolicy using CPU.
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        learning_rate=3e-4, 
        gamma=0.99, 
        verbose=1, 
        tensorboard_log=log_dir,
        device="cpu"
    )
    
    total_timesteps = 2000  # Adjust as needed.
    
    print("Starting training...")
    # Train the model with both callbacks.
    model.learn(
        total_timesteps=total_timesteps, 
        log_interval=10, 
        callback=[checkpoint_callback, print_callback]
    )
    
    # Save final model.
    model.save("ppo_drone_nav")
    print("Training complete. Model saved as 'ppo_drone_nav.zip'.")

if __name__ == "__main__":
    main()
