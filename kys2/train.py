# train.py
import argparse
from stable_baselines3 import DQN
from env import DroneEnv

def train_model(timesteps, model_save_path):
    # Create the environment without rendering for faster training.
    env = DroneEnv(render_mode=None)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drone RL Agent")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of timesteps for training")
    parser.add_argument("--save_path", type=str, default="dqn_drone", help="Path to save the trained model")
    args = parser.parse_args()
    train_model(args.timesteps, args.save_path)
