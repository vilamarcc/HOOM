from typing import Optional

import numpy as np

from gymnasium import spaces
import gymnasium as gym

import control as ctrl
from src.models.plane_model import plane_model
from src.utils.scales import angle_normalize

## ONLY LONGITUDINAL CHANNEL
class ControlEnv(gym.Env): 
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, plane_model: plane_model, deltae_max: float, deltaT_max: float, ref_x: list, render_mode: Optional[str] = None, dt: float = 0.1):
        super().__init__()

        self.plane_model = plane_model
        self.ref_x = ref_x
        self.state = np.array([0, 0, 0], dtype=np.float32)

        self.dt = dt

        self.max_deltae = deltae_max
        self.max_deltaT = deltaT_max

        self.render_mode = render_mode

        u_max = np.array([self.max_deltae, self.max_deltaT], dtype=np.float32)
        x_max = np.array([10, 1, 5], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-u_max, high=u_max, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

    def step(self, action):
        du, alpha, theta= self.state
        deltae, deltaT = action

        dt = self.dt

        u = [deltae, deltaT]
        x = [du, alpha, theta]
        u = np.clip(u, -self.max_deltae, self.max_deltae)
        self.last_u = u # for rendering

        #reward = -(0.1*angle_normalize(alpha - self.ref_x[1]) ** 2 
        #           + angle_normalize(theta  - self.ref_x[2]) ** 2 
        #           + 0.1*angle_normalize(q  - self.ref_x[3]) ** 2 
        #           + 0.001*(u - self.ref_x[0]))**2

        reward = -(0.1*angle_normalize(theta - self.ref_x[3])) ** 2

        A, B = self.plane_model.lon['A'], self.plane_model.lon['B']

        x_dot = np.dot(A, x) + np.dot(B, u)
        x_n = x + x_dot*dt

        if self.render_mode == "human":
            self.render()

        self.state = x_n

        return self._get_obs(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            x_max = np.array([10, 1, 5], dtype=np.float32)
 
        return self._get_obs(), {}
    
    def _get_obs(self):
        du, alpha, theta= self.state
        return np.array([du, alpha, theta])

    def render(self):
        #TODO
        return None

    def close(self):
        #TODO 
        return None