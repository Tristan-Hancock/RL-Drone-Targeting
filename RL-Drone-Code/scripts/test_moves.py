import airsim
import time

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Take off
    client.takeoffAsync().join()
    print("Took off.")

    # Ascend a bit
    client.moveByVelocityBodyFrameAsync(20, 0, -1, 30).join()
    print("Ascended.")

    # Move forward (drone's forward)
    client.moveByVelocityBodyFrameAsync(20, 0, 0, 30).join()
    print("Moved forward.")

    # Move right
    client.moveByVelocityBodyFrameAsync(0, 2, 0, 3).join()
    print("Moved right.")

    # Move backward
    client.moveByVelocityBodyFrameAsync(-2, 0, 0, 3).join()
    print("Moved backward.")

    # Descend
    client.moveByVelocityBodyFrameAsync(0, 0, 1, 3).join()
    print("Descended.")

    # Land
    client.landAsync().join()
    print("Landed.")

    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
