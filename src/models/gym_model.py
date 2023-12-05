from typing import Optional

import numpy as np
from sympy.abc import s, t

from gymnasium import spaces
import gymnasium as gym

import control as ctrl
import control.matlab as ctrlm
from src.models.plane_model import plane_model
from src.utils.scales import angle_normalize
from src.utils.conversion import to_time_domain
from src.services.flight_control import TimeVaryingStateSpace

## ONLY LONGITUDINAL CHANNEL
class ControlEnv(gym.Env): 
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, plane_model: plane_model, deltae_max: float, deltaT_max: float, ref_x: list, render_mode: Optional[str] = None, dt: float = 0.1):
        super().__init__()

        self.plane_model = plane_model
        self.ref_x = ref_x
        self.state = np.array([0, 0, 0, 0], dtype=np.float32)

        self.A = self.plane_model.lon['A']
        self.B = self.plane_model.lon['B']

        self.dt = dt

        self.max_deltae = deltae_max
        self.max_deltaT = deltaT_max

        self.sys = sys = ctrl.ss(self.A, self.B, np.eye(4), np.zeros((4, 2)), self.dt)

        self.xs, self.us = [], []
        self.ts = []
        self.ts.append(0)

        self.render_mode = render_mode

        u_max = np.array([self.max_deltae, self.max_deltaT])
        x_max = np.array([1., 1., 1., 1.])

        self.action_space = spaces.Box(
            low=-u_max, high=u_max, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

    def step(self, action):
        du, alpha, q, theta= self.state[0], self.state[1], self.state[2], self.state[3]
        deltae, deltaT = action

        dt = self.dt

        u = [deltae, deltaT]
        x = np.array([du, alpha, q, theta], dtype=np.float32)
        u[0] = np.clip(u[0], -self.max_deltae, self.max_deltae)
        u[1] = np.clip(u[1], -self.max_deltaT, self.max_deltaT)
        t_n = self.ts[-1]
        
        self.last_u = u # for rendering

        #reward = -(0.1*angle_normalize(alpha - self.ref_x[1]) ** 2 
        #           + angle_normalize(theta  - self.ref_x[2]) ** 2 
        #           + 0.1*angle_normalize(q  - self.ref_x[3]) ** 2 
        #           + 0.001*(u - self.ref_x[0]))**2

        reward = -(0.1*angle_normalize(theta - self.ref_x[3])) ** 2

        if reward == 0:
            done = True
        else:
            done = False
        
        x_dot = self.sys.dynamics(t_n, x, u)
        x_n = x + x_dot * dt

        if self.render_mode == "human":
            self.render()

        self.state = x_n
        self.us.append(u)
        self.xs.append(x_n)
        self.ts.append(t + dt)

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            x_max = np.array([1, 1, 1, 1], dtype=np.float32)
            self.xs, self.us, self.ts = [], [], []
            self.ts.append(0)
 
        return self._get_obs(), {}
    
    def _get_obs(self):
        du, alpha, q, theta = self.state
        return np.array([du, alpha, q, theta], dtype=np.float32)

    def render(self):
        #TODO
        return None

    def close(self):
        pass