import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from AirSimContinuousEnv import AirSimContinuousEnv

# Create the environment
env = AirSimContinuousEnv()

# Create a PPO model with tuned hyperparameters
model = PPO("MlpPolicy", env, verbose=1,
            n_steps=1024,            # rollout length
            learning_rate=3e-4,      # learning rate
            clip_range=0.2)          # clipping parameter

# Set up a callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='airsim_ppo')

# Train the model for 100,000 timesteps
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model
model.save("airsim_ppo_final")
