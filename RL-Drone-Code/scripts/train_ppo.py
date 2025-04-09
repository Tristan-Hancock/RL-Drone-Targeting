# train_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from DroneNavEnv import DroneNavEnv

# Create and wrap the environment to record training metrics
env = DroneNavEnv()
env = Monitor(env, filename="training_monitor.csv")

# Set up the PPO model with MLP policy and desired hyperparameters
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            clip_range=0.2,
            tensorboard_log="./ppo_tensorboard/")

# Callback to save the model every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_drone')

# Train the model for a total of 100,000 timesteps
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model
model.save("ppo_drone_final")

# Optionally, test the trained model
obs, _ = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
