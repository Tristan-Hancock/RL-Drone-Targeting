import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import os

# Import the updated environment
from Drone3DEnv import Drone3DEnv

def make_env(render_mode=None, num_buildings=10):
    """
    Create and wrap the Drone3D environment for better logging.
    This wrapper also supports the new Gymnasium API.
    """
    def _init():
        env = Drone3DEnv(render_mode=render_mode, num_buildings=num_buildings)
        env = Monitor(env)
        return env
    return _init

def train_agent(total_timesteps=200000, log_interval=10, save_freq=10000):
    """
    Train a PPO agent to navigate the drone through buildings to its base camp.
    """
    # Create output directories if they don't exist
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs/drone3d", exist_ok=True)
    
    # Create a vectorized training environment (set render_mode to None for faster training)
    vec_env = make_vec_env(make_env(render_mode=None, num_buildings=15), n_envs=4)
    
    # Create a separate environment for evaluation (no rendering during eval)
    eval_env = make_vec_env(make_env(render_mode=None, num_buildings=10), n_envs=1)
    
    # Create a callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./tb_logs/eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=True
    )
    
    # Create a checkpoint callback for periodic saving
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/checkpoints/",
        name_prefix="drone_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Create the PPO agent with your chosen hyperparameters
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        tensorboard_log="./tb_logs/drone3d/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,  # Encourage exploration
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
    )
    
    # If a previously saved model exists, load it to continue training.
    model_path = "./models/ppo_drone3d.zip"
    if os.path.exists(model_path):
        print("Loading existing model to continue training...")
        model = PPO.load(model_path, env=vec_env)
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=log_interval
    )
    
    # Save the final model
    model.save("./models/ppo_drone3d")
    
    vec_env.close()
    eval_env.close()
    
    print("Training complete and model saved.")
    return model

def evaluate_agent(model_path="./models/ppo_drone3d.zip", episodes=10):
    """
    Evaluate a trained agent with rendering enabled.
    """
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create the environment with rendering turned on
    env = Drone3DEnv(render_mode="human", num_buildings=10)
    
    total_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated  # Combine terminated and truncated flags
        
        total_rewards.append(episode_reward)
        success_count += terminated  # Count if the goal was reached (terminated flag)
        print(f"Episode {episode+1} - Reward: {episode_reward:.2f} - Success: {terminated}")
    
    print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Success rate: {success_count/episodes * 100:.1f}%")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate a drone navigation agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "both"],
                        help="Mode: 'train', 'eval', or 'both'")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total timesteps for training")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        model = train_agent(total_timesteps=args.timesteps)
    
    if args.mode in ["eval", "both"]:
        evaluate_agent(episodes=args.eval_episodes)
