#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_monitor_data(csv_path):
    """
    Loads monitor data from a CSV file.
    The Monitor CSV file generated by Stable Baselines3 typically
    has a header line that starts with '#'. The pandas read_csv() function
    will ignore those if we specify skiprows appropriately.
    """
    try:
        # Skip the first row if it starts with a '#'
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        skip_rows = 1 if first_line.startswith('#') else 0
        
        data = pd.read_csv(csv_path, skiprows=skip_rows)
        return data
    except Exception as e:
        print(f"Error loading monitor data from {csv_path}: {e}")
        return None

def plot_metrics(data):
    """
    Plots evaluation metrics from monitor CSV data.
    """
    # Create cumulative rewards column.
    data['cumulative'] = data['r'].cumsum()
    
    plt.figure(figsize=(12, 10))
    
    # Plot Episode Reward per Episode.
    plt.subplot(2, 2, 1)
    plt.plot(data['r'], marker='o', linestyle='-', color='blue')
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot Episode Length per Episode.
    plt.subplot(2, 2, 2)
    plt.plot(data['l'], marker='o', linestyle='-', color='green')
    plt.title("Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    
    # Plot Cumulative Reward over Episodes.
    plt.subplot(2, 2, 3)
    plt.plot(data['cumulative'], marker='o', linestyle='-', color='magenta')
    plt.title("Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    
    # Plot Histogram of Episode Rewards.
    plt.subplot(2, 2, 4)
    plt.hist(data['r'], bins=20, color='orange', edgecolor='black')
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO training metrics from Monitor CSV")
    parser.add_argument(
        "--csv", type=str, default="./tb_logs/drone3d/monitor.csv",
        help="Path to the Monitor CSV file (default: ./tb_logs/drone3d/monitor.csv)"
    )
    args = parser.parse_args()
    
    data = load_monitor_data(args.csv)
    if data is not None:
        print("Columns in Monitor CSV:", data.columns.tolist())
        plot_metrics(data)
    else:
        print("No data available to plot.")

if __name__ == "__main__":
    main()
