# record_flight_path.py
import matplotlib.pyplot as plt
import numpy as np
from DroneNavEnv import DroneNavEnv

env = DroneNavEnv()
obs = env.reset()

done = False
while not done:
    # For testing, use a random policy (replace with model.predict() for trained policy)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

# Convert trajectory list to NumPy array for plotting
trajectory = np.array(env.trajectory)
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label="Flight Path")
plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], c='green', marker='o', s=100, label="Start")
plt.scatter([env.goal_pos[0]], [env.goal_pos[1]], c='red', marker='X', s=100, label="Goal")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Drone Flight Path (Top-Down View)")
plt.legend()
plt.grid(True)
plt.savefig("flight_path.png")
plt.show()
env.close()
