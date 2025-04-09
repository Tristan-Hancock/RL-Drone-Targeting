# visualize_training.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the monitor file (created by the Monitor wrapper)
data = pd.read_csv("training_monitor.csv", comment='#')

# Plot episode rewards over time
plt.figure(figsize=(10, 5))
plt.plot(data['r'], label='Episode Reward')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("training_reward_plot.png")  # Save the figure
plt.show()
