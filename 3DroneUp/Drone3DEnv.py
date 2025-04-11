import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import random
import math

try:
    from ursina import Ursina, Entity, Sky, color, camera, Vec3, window, application
except ImportError:
    raise ImportError("Please install ursina via 'pip install ursina' to enable 3D rendering.")

TRAINING_MODE = True

class Drone3DEnv(gym.Env):
    """
    A 3D drone navigation environment with fixed buildings.

    Key improvements:
      1. The drone's velocity is now included in the observation, so the policy can learn how
         its movement evolves over time.
      2. An incremental distance reward is used, giving a small bonus each step the drone
         reduces its distance to the goal, encouraging consistent progress.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, vehicle_name="Drone1"):
        super().__init__()
        self.render_mode = render_mode
        self.vehicle_name = vehicle_name

        # Simulation/time constraints.
        self.dt = 0.1
        self.max_steps = 300
        self.current_step = 0

        # Reward/Penalty parameters.
        self.collision_penalty = -20.0      # Extra harsh collision penalty.
        self.time_penalty = -0.05           # Lower step penalty than before.
        self.success_bonus  = 300.0         # Reaching the goal yields big reward.
        # Add a small distance-improvement factor to encourage moving closer to the goal each step.
        self.incremental_distance_factor = 0.05

        # Boundaries (z=0 is ground).
        self.boundary_min = np.array([-50, -50, 0])
        self.boundary_max = np.array([50, 50, 30])

        # Action space: 3D continuous command.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # ---------------------------
        # Enhanced Observation Space
        # ---------------------------
        # We'll include:
        #   1) Drone position (3),
        #   2) Drone velocity (3),
        #   3) 8 sensor readings  => total dimension = 14.
        self.sensor_count = 8
        self.sensor_range = 100.0
        obs_dim = 3 + 3 + self.sensor_count  # position + velocity + sensors = 14

        # Lower bounds: [boundary_min, -max_velocity, 0 for sensors].
        # Upper bounds: [boundary_max, +max_velocity, sensor_range for sensors].
        obs_low = np.concatenate([
            self.boundary_min,  # position
            -np.ones(3, dtype=np.float32) *  self.boundary_max[0],  # velocity (just assume –50..+50 is safe)
            np.zeros(self.sensor_count, dtype=np.float32)
        ])
        obs_high = np.concatenate([
            self.boundary_max,  # position
            np.ones(3, dtype=np.float32) *  self.boundary_max[0],   # velocity (just assume 50.. is safe)
            np.ones(self.sensor_count, dtype=np.float32) * self.sensor_range
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Start & goal positions, velocity.
        self.drone_start = np.array([-40.0, -40.0, 15.0])
        self.goal_position = np.array([40.0, 40.0, 2.0])
        self.goal_threshold = 3.0
        self.max_velocity = 0.8

        self.state = self.drone_start.copy()
        self.velocity = np.zeros(3)
        self.in_collision = False

        # We'll keep track of last-step distance to compute incremental improvement.
        self.prev_distance = np.linalg.norm(self.state - self.goal_position)

        # Building setup.
        self.buildings = []
        self.building_positions = []
        self.building_sizes = []
        self.building_colors = [
            color.light_gray, color.gray, color.dark_gray, color.cyan, color.blue, color.brown
        ]
        self._generate_fixed_buildings()

        if self.render_mode is not None:
            self.setup_ursina_environment()
        else:
            self.app = None

    def _generate_fixed_buildings(self):
        """
        Define a fixed layout of buildings to block the direct path.
        Adjust positions/sizes to your preference. 
        """
        fixed_layout = [
            {"position": np.array([-20, -20, 7.5]), "scale": (8, 8, 15)},
            {"position": np.array([-5, -5, 10]),    "scale": (10, 10, 20)},
            {"position": np.array([10, 0, 6]),      "scale": (6, 6, 12)},
            {"position": np.array([0, 20, 8]),      "scale": (8, 8, 16)},
            {"position": np.array([20, 20, 6]),     "scale": (6, 6, 12)},
        ]
        for b in fixed_layout:
            pos = b["position"]
            scale = b["scale"]
            self.building_positions.append(tuple(pos))
            self.building_sizes.append(scale)
            self.buildings.append(None)

    def setup_ursina_environment(self):
        from ursina import Ursina, Entity, Sky, color, camera, Vec3, window, application

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

        # Sky
        self.sky = Sky(texture="Assets/skyblue.jpg")

        # Drone
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

        # Goal
        self.goal_entity = Entity(
            model='quad',
            texture='Assets/goal2.png',
            scale=6.0,
            position=Vec3(*self.goal_position),
            billboard=True,
            color=color.white,
            transparency=True
        )
        self.goal_entity.animate_rotation_y(360, duration=4, loop=True)

        # Create building entities from the stored positions.
        from ursina import Entity
        new_buildings = []
        for pos, size in zip(self.building_positions, self.building_sizes):
            pos_arr = np.array(pos)
            building = Entity(
                model='cube',
                color=random.choice(self.building_colors),
                position=Vec3(*pos_arr),
                scale=size,
                texture='brick',
                collider='box'
            )
            new_buildings.append(building)
        self.buildings = new_buildings

        # Camera
        from ursina import camera
        camera.position = Vec3(self.drone_start[0], self.drone_start[1] - 10, self.drone_start[2] + 5)
        camera.look_at(self.drone_entity)

    def compute_sensor_readings(self):
        """
        Compute 8 sensor readings in evenly spaced directions (horizontal plane).
        """
        readings = np.ones(self.sensor_count, dtype=np.float32) * self.sensor_range
        for i in range(self.sensor_count):
            angle = 2 * math.pi * i / self.sensor_count
            sensor_dir = np.array([math.cos(angle), math.sin(angle)])
            min_dist = self.sensor_range
            for pos, size in zip(self.building_positions, self.building_sizes):
                pos_arr = np.array(pos)
                rel = pos_arr[:2] - self.state[:2]
                dist = np.linalg.norm(rel)
                if dist < 1e-3:
                    continue
                rel_norm = rel / dist
                # within ~20° of the sensor direction
                if np.dot(sensor_dir, rel_norm) > 0.94:
                    building_radius = (size[0] + size[1]) / 4.0
                    effective_dist = max(0, dist - building_radius)
                    min_dist = min(min_dist, effective_dist)
            readings[i] = min_dist
        return readings

    def compute_proximity_penalty(self):
        """
        Returns a negative penalty if the drone is too close to any building.
        """
        penalty = 0.0
        safe_distance = 5.0
        for pos, size in zip(self.building_positions, self.building_sizes):
            pos_arr = np.array(pos)
            d = np.linalg.norm(self.state[:2] - pos_arr[:2])
            building_radius = (size[0] + size[1]) / 4.0
            if d < (safe_distance + building_radius):
                penalty += (safe_distance + building_radius - d) * 0.5
        return -penalty

    def _get_observation(self):
        """
        Return [drone_position (3), drone_velocity (3), sensor_readings (8)] => total 14.
        """
        sensors = self.compute_sensor_readings()
        return np.concatenate((self.state, self.velocity, sensors)).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.current_step = 0
        self.in_collision = False

        self.state = self.drone_start.copy()
        self.velocity = np.zeros(3)
        self.prev_distance = np.linalg.norm(self.state - self.goal_position)

        if self.render_mode is not None:
            self.drone_entity.position = Vec3(*self.state)
            from ursina import camera
            camera.position = Vec3(self.state[0], self.state[1] - 10, self.state[2] + 5)
            camera.look_at(self.drone_entity)
            self.app.step()
        return self._get_observation(), {}

    def check_collision(self):
        """
        True if the drone collides with a building or out-of-bounds.
        """
        drone_pos = self.state
        # Buildings
        for pos, size in zip(self.building_positions, self.building_sizes):
            pos_arr = np.array(pos)
            half_dims = np.array(size) / 2.0
            if (abs(drone_pos[0] - pos_arr[0]) < half_dims[0] and
                abs(drone_pos[1] - pos_arr[1]) < half_dims[1] and
                abs(drone_pos[2] - pos_arr[2]) < half_dims[2]):
                return True
        # Boundaries
        if (drone_pos[0] < self.boundary_min[0] or drone_pos[0] > self.boundary_max[0] or
            drone_pos[1] < self.boundary_min[1] or drone_pos[1] > self.boundary_max[1] or
            drone_pos[2] < self.boundary_min[2] or drone_pos[2] > self.boundary_max[2]):
            return True
        return False

    def step(self, action):
        self.current_step += 1

        # Process action.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        acceleration = action * self.max_velocity * 0.5
        # Momentum-based velocity update
        self.velocity = 0.9 * self.velocity + acceleration
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        old_state = self.state.copy()
        self.state += self.velocity

        # Clamp state to boundaries
        self.state = np.clip(self.state, self.boundary_min, self.boundary_max)

        # Collision check
        self.in_collision = self.check_collision()
        collision_reward = 0.0
        if self.in_collision:
            self.state = old_state
            self.velocity *= -0.5
            collision_reward = self.collision_penalty

        # Distance calculations
        distance = np.linalg.norm(self.state - self.goal_position)
        # Base distance penalty
        distance_reward = -0.2 * distance  # more emphasis

        # Incremental reward for moving closer
        incremental_improvement = (self.prev_distance - distance)
        inc_reward = self.incremental_distance_factor * incremental_improvement
        self.prev_distance = distance

        # Proximity penalty
        proximity_penalty = self.compute_proximity_penalty()

        total_reward = distance_reward + collision_reward + self.time_penalty + proximity_penalty + inc_reward

        # If we reached goal
        terminated = bool(distance < self.goal_threshold)
        if terminated:
            total_reward += self.success_bonus

        truncated = bool(self.current_step >= self.max_steps)
        info = {
            "step": self.current_step,
            "distance": distance,
            "collision": self.in_collision
        }

        # Rendering
        if self.render_mode is not None:
            self.drone_entity.position = Vec3(*self.state)
            from ursina import camera
            target_cam_pos = Vec3(self.state[0], self.state[1] - 10, self.state[2] + 5)
            camera.position = camera.position + (target_cam_pos - camera.position) * 0.1
            camera.look_at(self.drone_entity)
            self.drone_entity.color = color.red if self.in_collision else color.white
            self.app.step()

        return self._get_observation(), total_reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.render_mode is None:
            return None
        if mode == "human":
            return None
        elif mode == "rgb_array":
            return np.zeros((600,600,3), dtype=np.uint8)

    def close(self):
        print(f"[CLOSE] Disarming and resetting simulation for {self.vehicle_name}...")
        if self.app is not None:
            application.quit()


# For testing/demo:
if __name__ == "__main__":
    env = Drone3DEnv(render_mode="human")
    obs, _ = env.reset()
    print("Initial observation dimension:", len(obs))
    print("Goal position:", env.goal_position)
    print("Navigate the city to reach your base camp!")

    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample() * 0.3
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {info['step']} - Reward: {reward:.2f}, Dist: {info['distance']:.2f}, Collision: {info['collision']}")
        done = terminated or truncated
        time.sleep(0.05)

    if terminated:
        print(f"Success! Reached base camp. Total reward: {total_reward:.2f}")
    else:
        print(f"Failed. Total reward: {total_reward:.2f}")

    env.close()
