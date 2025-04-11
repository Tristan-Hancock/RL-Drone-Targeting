# env.py
import gym
from gym import spaces
import numpy as np
import pygame
import os

class DroneEnv(gym.Env):
    """
    A custom Gym environment for a drone landing simulation.
    The drone (represented by drone.png) starts at a fixed location and
    must reach the landing pad (represented by landing.png).
    When the drone reaches the goal area, the environment prints and displays "Drone Landed".
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()
        self.render_mode = render_mode

        # Define window dimensions
        self.window_width = 600
        self.window_height = 600

        # Simulation parameters
        self.max_steps = 200
        self.step_count = 0
        self.drone_speed = 5  # Pixels per action

        # Define discrete action space: 0: right, 1: left, 2: up, 3: down
        self.action_space = spaces.Discrete(4)
        # Observation space: drone's (x, y) position (floats)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([self.window_width, self.window_height]), 
            dtype=np.float32
        )

        # Initial drone position (bottom-left of the window)
        self.drone_pos = np.array([50, self.window_height - 50], dtype=np.float32)
        # Goal: landing pad location (placed toward the top-right)
        self.goal_pos = np.array([self.window_width - 100, 100], dtype=np.float32)
        # Flag to indicate if drone has landed
        self.landed = False

        # Initialize Pygame if rendering is enabled
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Drone Simulation")
            self.clock = pygame.time.Clock()
            # Load images; assumed to be located in the same directory.
            base_path = os.path.dirname(__file__)
            self.drone_img = pygame.image.load(os.path.join(base_path, "Assets/drone.png"))
            self.landing_img = pygame.image.load(os.path.join(base_path, "Assets/goal2.png"))
            # Optionally scale images to an appropriate size
            self.drone_img = pygame.transform.scale(self.drone_img, (50, 50))
            self.landing_img = pygame.transform.scale(self.landing_img, (80, 80))
            # Initialize a font for rendering text
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 30)

    def step(self, action):
        self.step_count += 1

        # Update the drone's position based on the action selected.
        if action == 0:  # right
            self.drone_pos[0] += self.drone_speed
        elif action == 1:  # left
            self.drone_pos[0] -= self.drone_speed
        elif action == 2:  # up
            self.drone_pos[1] -= self.drone_speed  # In Pygame, y decreases upward
        elif action == 3:  # down
            self.drone_pos[1] += self.drone_speed

        # Keep the drone within the window bounds
        self.drone_pos[0] = np.clip(self.drone_pos[0], 0, self.window_width - 50)
        self.drone_pos[1] = np.clip(self.drone_pos[1], 0, self.window_height - 50)

        # Compute the reward as the negative Euclidean distance to the goal.
        distance = np.linalg.norm(self.drone_pos - self.goal_pos)
        reward = -distance

        # Check if the drone has reached the goal (within 50 pixels)
        done = distance < 50 or self.step_count >= self.max_steps
        if distance < 50 and not self.landed:
            self.landed = True
            print("Drone Landed")

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.drone_pos.copy()

    def reset(self):
        self.drone_pos = np.array([50, self.window_height - 50], dtype=np.float32)
        self.step_count = 0
        self.landed = False
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        return self._get_obs()

    def render(self, mode='human'):
        if self.render_mode != "human":
            return

        # Fill the background (white)
        self.screen.fill((255, 255, 255))

        # Draw the landing pad (goal)
        landing_rect = self.landing_img.get_rect(center=(self.goal_pos[0], self.goal_pos[1]))
        self.screen.blit(self.landing_img, landing_rect)

        # Draw the drone at its current position
        drone_rect = self.drone_img.get_rect(topleft=(self.drone_pos[0], self.drone_pos[1]))
        self.screen.blit(self.drone_img, drone_rect)

        # If the drone has landed, render the "Drone Landed" text
        if self.landed:
            text_surface = self.font.render("Drone Landed", True, (0, 128, 0))
            # Position the text at the center of the screen
            text_rect = text_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))
            self.screen.blit(text_surface, text_rect)

        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
