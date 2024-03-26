import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MicroExpressionEnvironment(gym.Env):
    
    def __init__(self, data_path, image_size = (224, 224), binary_reward = False, big_reward = 1000, penalty = -500):
        super().__init__()
        
        self.data_path = data_path
        self.image_size = image_size
        self.binary_reward = binary_reward
        self.big_reward = big_reward
        self.penalty = penalty
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (*image_size, 1), dtype = np.uint8)
        
        self.episodes = self._load_episodes()
        self.current_episode_index = np.random.randint(len(self.episodes))
        self.current_episode = self._get_frames()
        self.current_frame_index = 0
        self.max_steps = 50
        self.current_step = 0
        
    def _get_frames(self):
        frames = []
        for frame in os.listdir(self.episodes[self.current_episode_index]):
            frame_path = os.path.join(self.episodes[self.current_episode_index], frame)
            frames.append(frame_path)
        return frames
    
    def _load_episodes(self):
        episodes = []
        for expression in os.listdir(self.data_path):
            expression_path = os.path.join(self.data_path, expression)
            for episode in os.listdir(expression_path):
                episode_path = os.path.join(expression_path, episode)
                episodes.append(episode_path)
        return episodes
    
    def _next_episode(self):
        self.current_episode_index = np.random.randint(len(self.episodes))
        self.current_episode = self._get_frames()
        self.current_frame_index = 0
    
    def _read_frame(self):
        frame_path = self.current_episode[self.current_frame_index]
        if not os.path.exists(frame_path):
            print(frame_path)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        return frame
    
    def _calculate_reward(self, action):
        
        if action == 0: # halt
            if self.current_frame_index == len(self.current_episode) - 1:
                return self.big_reward
            else:
                return self.penalty * abs(len(self.current_episode) - 1 - self.current_frame_index)
        elif action == 1: # continue
            if self.current_frame_index < len(self.current_episode) - 1:
                if self.binary_reward:
                    return -1
                else:
                    return -10 * (len(self.current_episode) - self.current_frame_index)
            else:
                return self.penalty
        
        
    def step(self, action):
        
        reward = self._calculate_reward(action)
        terminated = False
        truncated = False
        
        if action == 0: # halt
            if self.current_frame_index == len(self.current_episode) - 1:
                terminated = True
            # else:
            #     truncated = True
        else: # continue
            if self.current_frame_index < len(self.current_episode) - 1:
                self.current_frame_index += 1
            elif self.current_frame_index == len(self.current_episode) - 1:
                self.current_frame_index = len(self.current_episode) - 1
        self.current_step += 1
        
        # if terminated:
        #     self._next_episode()
        
        if self.current_step >= self.max_steps:
            truncated = True
            # if truncated:
            #     self._next_episode()
        
        observation = self._read_frame()
        return observation, reward, terminated, truncated
        
    def reset(self):
        self._next_episode()
        self.current_frame_index = 0
        self.current_step = 0
        return self._read_frame()
        
    def render(self, mode = 'human'):
        frame = self._read_frame(self.current_frame_index)
        if mode == 'human':
            cv2.imshow("MicroExpressionEnvironment", frame)
            cv2.waitKey(1)
        return frame
