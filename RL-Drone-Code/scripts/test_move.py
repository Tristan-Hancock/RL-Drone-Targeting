# test_move_to_altitude.py
import airsim
import time
import numpy as np

def main():
    client = airsim.MultirotorClient()
    print("Connecting to AirSim...")
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("Resetting simulation...")
    client.reset()
    time.sleep(3)
    
    print("Re-enabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(3)
    
    # Desired safe altitude in NED (negative values mean higher altitude)
    target_altitude = -10.0

    # Loop until the drone reaches the target altitude or a max number of iterations is hit.
    max_attempts = 10
    attempt = 0
    while attempt < max_attempts:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_alt = pos.z_val  # In NED, more negative means higher altitude.
        print(f"Attempt {attempt+1}: Current altitude = {current_alt:.2f}, target = {target_altitude}")
        # Check if altitude difference is within an acceptable tolerance.
        if np.abs(current_alt - target_altitude) < 0.5:
            break
        # Command the drone to move to the target altitude.
        client.moveToZAsync(target_altitude, 2).join()
        time.sleep(2)
        attempt += 1

    # Finally, hover and wait a bit.
    client.hoverAsync().join()
    time.sleep(2)
    
    # Print the final drone position.
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"Final Drone Position: X = {pos.x_val:.2f}, Y = {pos.y_val:.2f}, Z = {pos.z_val:.2f}")
    
    # Cleanup
    client.armDisarm(False)
    client.enableApiControl(False)
    client.reset()

if __name__ == "__main__":
    main()
