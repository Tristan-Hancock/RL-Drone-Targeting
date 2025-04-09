# test_goal_marker.py
import airsim
import time

def main():
    # Create AirSim client and connect.
    client = airsim.MultirotorClient()
    print("Connecting to AirSim...")
    client.confirmConnection()
    time.sleep(1)  # Give a moment for connection to settle

    # Attempt to retrieve the goal marker's pose.
    print("Requesting GoalMarker pose from the simulator...")
    goal_pose = client.simGetObjectPose("GoalMarker")

    # Check if the pose is valid (AirSim returns NaN if the object isn't found).
    pos = goal_pose.position
    if any(np.isnan([pos.x_val, pos.y_val, pos.z_val])):
        print("GoalMarker not found. Please ensure it exists in the Unreal level with the proper name or tag.")
    else:
        print("GoalMarker Position:")
        print(f"  X: {pos.x_val}")
        print(f"  Y: {pos.y_val}")
        print(f"  Z: {pos.z_val}")

    # Optionally, also print the drone's current position.
    drone_state = client.getMultirotorState()
    drone_pos = drone_state.kinematics_estimated.position
    print("\nDrone's Current Position:")
    print(f"  X: {drone_pos.x_val}")
    print(f"  Y: {drone_pos.y_val}")
    print(f"  Z: {drone_pos.z_val}")

if __name__ == "__main__":
    # Import numpy for NaN checking.
    import numpy as np
    main()
