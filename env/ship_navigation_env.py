import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class ShipNavigationEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.grid_size = 10
        self.num_obstacles = 6

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32
        )

        self.max_steps = 180

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.ship_pos = np.array([0, 0], dtype=float)

        # random goal
        while True:
            goal = np.random.randint(2, self.grid_size - 1, size=2).astype(float)
            if np.linalg.norm(goal - self.ship_pos) > 5:
                break

        self.goal_pos = goal
        self.current_step = 0

        # ocean current (slightly reduced randomness)
        self.current = random.choice([
            np.array([0,0]),
            np.array([1,0]),
            np.array([-1,0]),
            np.array([0,1]),
            np.array([0,-1])
        ])

        self.obstacles = []

        while len(self.obstacles) < self.num_obstacles:

            pos = np.random.randint(1, self.grid_size-1, size=2).astype(float)

            if any(np.array_equal(pos,o["pos"]) for o in self.obstacles):
                continue

            if np.array_equal(pos,self.ship_pos):
                continue

            if np.array_equal(pos,self.goal_pos):
                continue

            if np.linalg.norm(pos-self.ship_pos) < 3:
                continue

            direction = random.choice([
                np.array([1,0]),
                np.array([-1,0]),
                np.array([0,1]),
                np.array([0,-1])
            ])

            self.obstacles.append({
                "pos":pos,
                "dir":direction
            })

        return self._get_state(), {}

    def move_obstacles(self):

        for obs in self.obstacles:

            # small randomness in direction
            if random.random() < 0.08:
                obs["dir"] = random.choice([
                    np.array([1,0]),
                    np.array([-1,0]),
                    np.array([0,1]),
                    np.array([0,-1])
                ])

            obs["pos"] += obs["dir"]

            if obs["pos"][0] <= 0 or obs["pos"][0] >= self.grid_size-1:
                obs["dir"][0] *= -1

            if obs["pos"][1] <= 0 or obs["pos"][1] >= self.grid_size-1:
                obs["dir"][1] *= -1

            obs["pos"] = np.clip(obs["pos"],0,self.grid_size-1)

    def apply_current(self):

        if random.random() < 0.15:

            new_pos = self.ship_pos + self.current
            self.ship_pos = np.clip(new_pos,0,self.grid_size-1)

    def _normalize(self,val):

        return (val/(self.grid_size-1))*2 - 1

    def _get_state(self):

        dx = self.goal_pos[0] - self.ship_pos[0]
        dy = self.goal_pos[1] - self.ship_pos[1]

        obstacle_positions = [obs["pos"] for obs in self.obstacles]

        raw = np.concatenate((
            self.ship_pos,
            self.goal_pos,
            np.array([dx,dy]),
            *obstacle_positions
        ))

        if raw.shape[0] < 20:
            raw = np.pad(raw,(0,20-raw.shape[0]))

        return self._normalize(raw).astype(np.float32)

    def step(self,action):

        done=False
        self.current_step+=1

        self.move_obstacles()

        old_distance = np.linalg.norm(self.ship_pos-self.goal_pos)

        # ship movement
        if action==0:
            self.ship_pos[1]+=1
        elif action==1:
            self.ship_pos[1]-=1
        elif action==2:
            self.ship_pos[0]-=1
        elif action==3:
            self.ship_pos[0]+=1

        self.ship_pos = np.clip(self.ship_pos,0,self.grid_size-1)

        self.apply_current()

        new_distance = np.linalg.norm(self.ship_pos-self.goal_pos)

        # smoother reward shaping
        reward = -0.04
        reward += (old_distance-new_distance)*2.2

        # discourage sailing close to rocks
        for obs in self.obstacles:
            dist = np.linalg.norm(self.ship_pos-obs["pos"])
            if dist < 1.5:
                reward -= 0.4

        # collision
        for obs in self.obstacles:
            if np.array_equal(self.ship_pos,obs["pos"]):
                reward = -45
                done=True
                break

        # goal reached
        if np.array_equal(self.ship_pos,self.goal_pos):
            reward = 150
            done=True

        # timeout
        if self.current_step >= self.max_steps:
            done=True
            reward -= 8

        return self._get_state(), reward, done, False, {}