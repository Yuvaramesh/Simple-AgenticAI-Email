import numpy as np
import pygame
import gym
from gym import spaces
import time
class GridWorldEnv(gym.Env):
    """Custom 5x5 Grid World Environment for Reinforcement Learning"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        
        # Define the grid world dimensions
        self.grid_size = 5
        self.window_size = 400  # Size of the Pygame window
        
        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (agent's position)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        # Initialize Pygame for rendering
        pygame.init()
        self.screen = None
        self.cell_size = self.window_size // self.grid_size
        
        # Define rewards
        self.step_reward = -0.1  # Small penalty for each step
        self.goal_reward = 10.0   # Reward for reaching the goal
        self.wall_penalty = -1.0  # Penalty for hitting a wall
        
        # Define special positions
        self.goal_pos = np.array([4, 4])  # Bottom-right corner
        self.wall_pos = np.array([2, 2])  # Wall in the center
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        # Start position (top-left corner)
        self.agent_pos = np.array([0, 0])
        return self.agent_pos.copy()
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Store previous position
        prev_pos = self.agent_pos.copy()
        
        # Move the agent according to the action
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 3:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        
        # Calculate reward
        reward = self.step_reward
        
        # Check if agent hit the wall
        if np.array_equal(self.agent_pos, self.wall_pos):
            reward += self.wall_penalty
            self.agent_pos = prev_pos  # Revert position
        
        # Check if agent reached the goal
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward += self.goal_reward
        
        # Additional info (can be used for debugging)
        info = {}
        
        return self.agent_pos.copy(), reward, done, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        
        # Create a surface if we need to return an RGB array
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        
        # Draw grid lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (0, 0, 0),
                (0, x * self.cell_size),
                (self.window_size, x * self.cell_size),
                width=1
            )
            pygame.draw.line(
                canvas, (0, 0, 0),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size),
                width=1
            )
        
        # Draw wall
        wall_rect = pygame.Rect(
            self.wall_pos[1] * self.cell_size,
            self.wall_pos[0] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(canvas, (139, 69, 19), wall_rect)  # Brown wall
        
        # Draw goal
        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size,
            self.goal_pos[0] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(canvas, (0, 255, 0), goal_rect)  # Green goal
        
        # Draw agent
        agent_center = (
            int((self.agent_pos[1] + 0.5) * self.cell_size),
            int((self.agent_pos[0] + 0.5) * self.cell_size)
        )
        pygame.draw.circle(
            canvas, (255, 0, 0),  # Red agent
            agent_center, int(self.cell_size / 3)
        )
        
        if mode == 'human':
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment and Pygame window"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Example usage
if __name__ == "__main__":
    env = GridWorldEnv()
    
    # Test the environment with random actions
    obs = env.reset()
    env.render()
    
    for _ in range(20):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        
        env.render()
        step_delay = 0.5 

        print(f"Action: {action}, New Position: {obs}, Reward: {reward}, Done: {done}")
        time.sleep(step_delay)

        if done:
            print("Goal reached!")
            break
    
    env.close()