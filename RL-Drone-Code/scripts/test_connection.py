import airsim
import time

def main():
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Take off the drone
    client.takeoffAsync().join()
    print("Drone has taken off.")
    
    # Retrieve and print the drone's current state
    state = client.getMultirotorState()
    print("Current Position:", state.kinematics_estimated.position)
    
    # Simple flight command: move forward
    client.moveByVelocityAsync(1, 0, 10, 10).join()
    print("Moved forward for 3 seconds.")
    
    # Land the drone
    client.landAsync().join()
    print("Drone has landed.")
    
    # Disarm and reset control
    client.armDisarm(False)
    client.reset()
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
