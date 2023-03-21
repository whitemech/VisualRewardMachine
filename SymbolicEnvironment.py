import gym
from gym import spaces
import pygame
import random
import numpy as np


#tutta la griglia
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, moore_machine, render_mode = "human", train = True, size = 4):

        self._PICKAXE = "imgs/pickaxe.png"
        self._GEM = "imgs/gem.png"
        self._DOOR = "imgs/door.png"
        self._ROBOT = "imgs/robot.png"
        self._LAVA = "imgs/lava.jpg"
        self._train = train
        self.max_num_steps = 30
        self.curr_step = 0

        self.size = size #4x4 world
        self.window_size = 512 #size of the window
        
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

        self.automaton = moore_machine

        self.action_space = spaces.Discrete(4)
        # 0 = GO_DOWN
        # 1 = GO_RIGHT
        # 2 = GO_UP
        # 3 = GO_LEFT
        self.observation_space = spaces.MultiDiscrete([4,4,2,2,2,2,2,2])

        self._action_to_direction = {
            0: np.array([0, 1]), #DOWN
            1: np.array([1, 0]), #RIGHT
            2: np.array([0, -1]), #UP
            3: np.array([-1, 0]), #LEFT
        }

    def reset(self):
        '''
        TUTTO IL RESET 
        '''
        self.curr_step = 0
        self._agent_location = np.array([0, 0])
        self._gem_location = np.array([0, 3])
        self._pickaxe_location = np.array([1, 1])
        self._exit_location = np.array([3, 0])
        self._lava_location = np.array([3, 3])
        self._state = np.array([0, 0, 0, 0, 1, 0])

        self._has_pickaxe = False
        self._has_gem = False
        self._went_into_lava = False
        self._task_completed = False

        self.r1_available = True
        self.r2_available = True
        self.r3_available = True
        self.r4_available = True

        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = False if self._train else True
        
        if self.render_mode == "human":
            self._render_frame()

        observation = self._agent_location
        observation = np.append(observation, self._state)
        info = self._get_info()
        reward = -3
        return observation

        
    def _check_door(self):
        if not self._went_into_lava and self._has_gem and self._has_pickaxe and (self._agent_location == self._exit_location).all():
            self._task_completed = True

    
    def _check_pickaxe(self):
        if not self._went_into_lava:
            if (self._agent_location == self._pickaxe_location).all():
                self._has_pickaxe = True
                # self._pickaxe_display = False

    def _check_gem(self):
        if not self._went_into_lava:
            if (self._agent_location == self._gem_location).all(): #and self._has_pickaxe:
                self._has_gem = True
                # self._gem_display = False

    def _check_lava(self):
        if (self._agent_location == self._lava_location).all():
            self._went_into_lava = True

    def step(self, action):
        '''
        TUTTI GLI STEP
        '''

        info = {'distance': 3}
        reward = -1
        self.curr_step += 1

        #MOVEMENT
        if action == 0:
            direction = np.array([0, 1])
        elif action == 1:
            direction = np.array([1, 0])
        elif action == 2:
            direction = np.array([0, -1])
        elif action == 3:
            direction = np.array([-1, 0])

        self._agent_location = np.clip(self._agent_location+direction, 0, self.size-1)
        observation = self._agent_location

        #check items
        #CHECK IN THIS ORDER
        self._check_lava()
        self._check_pickaxe()
        self._check_gem()
        self._check_door()

        #update automaton state
        if self._went_into_lava:
            self._state = np.array([0, 0, 0, 0, 0, 1])
            if self.r4_available:
                reward = -25
                self.r4_available = False
            info['distance'] = 4

        elif self._task_completed:
            self.state = np.array([1, 0, 0, 0, 0, 0])
            if self.r1_available:
                reward = 25
                self.r1_available = False
            info['distance'] = 0

        elif self._has_pickaxe and self._has_gem:
            self._state = np.array([0, 1, 0, 0, 0, 0])
            if self.r2_available or self.r3_available:
                reward = 10
                self.r2_available = False
                self.r3_available = False
            info['distance'] = 1

        elif self._has_pickaxe and not self._has_gem:
            self._state = np.array([0, 0, 1, 0, 0, 0])
            if self.r2_available:
                reward = 5
                self.r2_available = False
            info['distance'] = 2

        elif self._has_gem and not self._has_pickaxe:
            self._state = np.array([0, 0, 0, 1, 0, 0])
            if self.r3_available:
                reward = 5
                self.r3_available = False
            info['distance'] = 2

        elif not self._has_gem and not self._has_pickaxe:
            self._state = np.array([0, 0, 0, 0, 1, 0])

        if self.render_mode == "human":
                self._render_frame()

        observation = np.append(observation, self._state)

        done = self.curr_step >= self.max_num_steps

        # info = self._get_info()

        return observation, reward, done, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_obs(self):
        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1,0,2)
            )
        img = img[:,:,::-1]
        obs = img
        return obs

    def _get_info(self):
        info = {
            "robot location": self._agent_location,
            "inventory": "empty"
        }
        if self._has_gem:
            info["inventory"] = "gem"
        elif self._has_pickaxe: 
            info["inventory"] = "pickaxe"
        else:
            info["inventory"] = "empty"
        return info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size,self.window_size))
        canvas.fill((255,255,255))

        pix_square_size = (self.window_size/self.size)

        for x in range(self.size+1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        
        if self.render_mode == "human":
            pickaxe = pygame.image.load(self._PICKAXE)
            gem = pygame.image.load(self._GEM)
            door = pygame.image.load(self._DOOR)
            robot = pygame.image.load(self._ROBOT)
            lava = pygame.image.load(self._LAVA)
            self.window.blit(canvas, canvas.get_rect())

            if self._robot_display:
                self.window.blit(robot, (pix_square_size*self._agent_location[0],pix_square_size*self._agent_location[1]))
            if self._pickaxe_display:
                self.window.blit(pickaxe, (pix_square_size*self._pickaxe_location[0], pix_square_size*self._pickaxe_location[1]))
            if self._gem_display:
                self.window.blit(gem, (pix_square_size*self._gem_location[0],32+pix_square_size*self._gem_location[1]))
            self.window.blit(door, (pix_square_size*self._exit_location[0], pix_square_size*self._exit_location[1]))
            self.window.blit(lava, (pix_square_size*self._lava_location[0] + 2, pix_square_size*self._lava_location[1] + 2))
                
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
