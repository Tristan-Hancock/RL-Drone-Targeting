import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import random

# Import Ursina and related classes for 3D rendering.
try:
    from ursina import Ursina, Entity, Sky, color, camera, Vec3, window, application
except ImportError:
    raise ImportError("Please install ursina via 'pip install ursina' to enable 3D rendering.")

# Flag to indicate if we are in training mode.
TRAINING_MODE = True

class Drone3DEnv(gym.Env):
    """
    A 3D drone navigation environment simulating a city.
    The drone (rendered using 'drone.png') must navigate to its base camp (rendered using 'goal_flag.png')
    while avoiding randomly placed buildings. The background uses 'topview.png'.
    The state is the 3D position of the drone, the action is a 3D continuous command, and
    the reward is computed based on the negative distance to the goal along with collision and time penalties.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, num_buildings=10, vehicle_name="Drone1"):
        super().__init__()
        self.render_mode = render_mode
        self.num_buildings = num_buildings
        self.vehicle_name = vehicle_name

        # Simulation parameters
        self.dt = 0.1
        self.max_steps = 300
        self.current_step = 0
        self.collision_penalty = -10.0
        self.time_penalty = -0.1

        # Environment boundaries (city area).
        self.boundary_min = np.array([-50, -50, 0])  # ground at z=0
        self.boundary_max = np.array([50, 50, 30])

        # Define action and observation spaces.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.boundary_min, high=self.boundary_max, shape=(3,), dtype=np.float32)

        # Starting and goal positions.
        self.drone_start = np.array([-40.0, -40.0, 15.0])
        self.goal_position = np.array([40.0, 40.0, 2.0])  # landing spot on the ground.
        self.goal_threshold = 3.0
        self.max_velocity = 0.8

        # Drone state and collision tracking.
        self.state = self.drone_start.copy()
        self.velocity = np.zeros(3)
        self.in_collision = False

        # Lists for generated buildings.
        self.buildings = []
        self.building_positions = []
        self.building_sizes = []
        self.building_colors = [
            color.light_gray, color.gray, color.dark_gray,
            color.cyan, color.blue, color.brown, color.brown
        ]

        if self.render_mode is not None:
            self.setup_ursina_environment()
        else:
            self.app = None

    def setup_ursina_environment(self):
        self.app = Ursina(extra_optimization=True)
        window.title = f"City Drone Navigation - Vehicle: {self.vehicle_name}"
        window.borderless = False
        window.exit_button.visible = True

        # Ground plane.
        self.ground = Entity(
            model='plane',
            scale=(100, 1, 100),
            color=color.green,
            texture='grass',
            collision=True,
            position=Vec3(0, 0, 0)
        )

        # Sky background.
        self.sky = Sky(texture="Assets/grass.jpeg", color=color.clear)

        # Drone entity with a quad textured by drone.png.
        self.drone_entity = Entity(
            model='quad',
            texture='Assets/drone.png',
            scale=2.0,
            position=Vec3(*self.drone_start),
            billboard=True,
            color=color.white,
            transparency=True,
            collider='box'
        )

        # Goal flag entity using a quad textured by goal_flag.png.
        self.goal_entity = Entity(
            model='quad',
            texture='Assets/goal.png',
            scale=4.0,
            position=Vec3(*self.goal_position),
            billboard=True,
            color=color.white,
            transparency=True
        )
        self.goal_entity.animate_rotation_y(360, duration=4, loop=True)

        # Generate random buildings.
        self.generate_buildings()

        # Camera setup: follow the drone.
        camera.position = Vec3(self.drone_start[0], self.drone_start[1] - 10, self.drone_start[2] + 5)
        camera.look_at(self.drone_entity)

    def generate_buildings(self):
        # Clear previous buildings.
        for building in self.buildings:
            building.disable()
            building.visible = False

        self.buildings = []
        self.building_positions = []
        self.building_sizes = []

        # Generate buildings within boundaries, avoiding the start and goal.
        for _ in range(self.num_buildings):
            for _ in range(20):  # Try 20 times to place a building.
                width = random.uniform(3, 8)
                depth = random.uniform(3, 8)
                height = random.uniform(5, 20)
                pos_x = random.uniform(self.boundary_min[0] + 5, self.boundary_max[0] - 5)
                pos_y = random.uniform(self.boundary_min[1] + 5, self.boundary_max[1] - 5)
                position = np.array([pos_x, pos_y, height/2])
                if (np.linalg.norm(position[:2] - self.drone_start[:2]) > 12 and
                    np.linalg.norm(position[:2] - self.goal_position[:2]) > 12):
                    overlap = False
                    for other_pos, other_size in zip(self.building_positions, self.building_sizes):
                        dist = np.linalg.norm(position[:2] - other_pos[:2])
                        min_dist = (width + other_size[0]) / 2 + (depth + other_size[1]) / 2
                        if dist < min_dist:
                            overlap = True
                            break
                    if not overlap:
                        break
            building_color = random.choice(self.building_colors)
            building = Entity(
                model='cube',
                color=building_color,
                position=Vec3(*position),
                scale=(width, depth, height),
                texture='brick',
                collider='box'
            )
            self.buildings.append(building)
            self.building_positions.append(position)
            self.building_sizes.append((width, depth, height))

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.current_step = 0
        self.in_collision = False
        self.state = self.drone_start.copy()
        self.velocity = np.zeros(3)
        if self.render_mode is not None:
            self.generate_buildings()
            self.drone_entity.position = Vec3(*self.state)
            camera.position = Vec3(self.state[0], self.state[1] - 10, self.state[2] + 5)
            camera.look_at(self.drone_entity)
            self.app.step()
        return self.state.copy(), {}

    def check_collision(self):
        """Check if the drone collides with any building or exceeds the boundaries."""
        drone_pos = self.state
        for building_pos, building_size in zip(self.building_positions, self.building_sizes):
            width, depth, height = building_size
            half_width, half_depth, half_height = width / 2, depth / 2, height / 2
            if (abs(drone_pos[0] - building_pos[0]) < half_width and
                abs(drone_pos[1] - building_pos[1]) < half_depth and
                abs(drone_pos[2] - building_pos[2]) < half_height):
                return True
        if (drone_pos[0] < self.boundary_min[0] or drone_pos[0] > self.boundary_max[0] or
            drone_pos[1] < self.boundary_min[1] or drone_pos[1] > self.boundary_max[1] or
            drone_pos[2] < self.boundary_min[2] or drone_pos[2] > self.boundary_max[2]):
            return True
        return False

    def step(self, action):
        self.current_step += 1

        # Process action: clip and scale it.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        acceleration = action * self.max_velocity * 0.5
        self.velocity = 0.9 * self.velocity + acceleration  # dampened acceleration.
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        
        old_position = self.state.copy()
        self.state += self.velocity

        # Check for collisions.
        self.in_collision = self.check_collision()
        if self.in_collision:
            self.state = old_position
            self.velocity *= -0.5
            collision_reward = self.collision_penalty
        else:
            collision_reward = 0.0

        distance = np.linalg.norm(self.state - self.goal_position)
        distance_reward = -0.1 * distance
        total_reward = distance_reward + collision_reward + self.time_penalty

        terminated = bool(distance < self.goal_threshold)
        truncated = bool(self.current_step >= self.max_steps)
        info = {"step": self.current_step, "distance": distance, "collision": self.in_collision}

        if self.render_mode is not None:
            self.drone_entity.position = Vec3(*self.state)
            # Smooth camera follow via manual interpolation.
            target_cam_pos = Vec3(self.state[0], self.state[1] - 10, self.state[2] + 5)
            camera.position = camera.position + (target_cam_pos - camera.position) * 0.1
            camera.look_at(self.drone_entity)
            self.drone_entity.color = color.red if self.in_collision else color.white
            self.app.step()

        return self.state.copy(), total_reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.render_mode is None:
            return None
        if mode == "human":
            return None  # The Ursina window updates automatically.
        elif mode == "rgb_array":
            return np.zeros((600, 600, 3), dtype=np.uint8)  # Placeholder

    def close(self):
        print(f"[CLOSE] Disarming and resetting simulation for {self.vehicle_name}...")
        if self.app is not None:
            application.quit()

# For testing/demo purposes.
if __name__ == "__main__":
    env = Drone3DEnv(render_mode="human", num_buildings=15)
    obs, _ = env.reset()
    print("Initial observation:", obs)
    print("Goal position:", env.goal_position)
    print("Navigate the city to reach your base camp!")
    
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample() * 0.3  # Reduced action magnitude for smoother movement.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {info['step']} - Reward: {reward:.2f}, Distance: {info['distance']:.2f}, Collision: {info['collision']}")
        done = terminated or truncated
        time.sleep(0.05)
        
    if terminated:
        print(f"Success! Reached base camp with total reward: {total_reward:.2f}")
    else:
        print(f"Failed to reach base camp. Total reward: {total_reward:.2f}")
        
    env.close()
