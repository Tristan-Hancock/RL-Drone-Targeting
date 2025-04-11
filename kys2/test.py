# test.py
import time
import argparse
import pygame
from stable_baselines3 import DQN
from env import DroneEnv

def test_model(model_path):
    # Create the environment with rendering enabled.
    env = DroneEnv(render_mode="human")
    model = DQN.load(model_path, env=env)
    obs = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)  # Slow down for visualization

    # Pause to allow the user to see "Drone Landed" on the screen if landed,
    # then exit when the window is closed.
    if env.landed:
        print("Drone Landed")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        env.close()
    else:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Drone RL Agent")
    parser.add_argument("--load_path", type=str, default="dqn_drone", help="Path to the saved model")
    args = parser.parse_args()
    test_model(args.load_path)
