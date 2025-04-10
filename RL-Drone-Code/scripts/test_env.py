from DroneNavEnv import DroneNavEnv
import time

def main():
    env = DroneNavEnv()
    obs, _ = env.reset()
    print("Initial observation:", obs)
    # Execute one random action manually
    action = env.action_space.sample()
    print("Taking action:", action)
    obs, reward, terminated, truncated, _ = env.step(action)
    print("After 1 step:")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated, "Truncated:", truncated)
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    main()
