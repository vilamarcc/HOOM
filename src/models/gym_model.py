from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import s, t

from gymnasium import spaces
import gymnasium as gym

import control as ctrl
import control.matlab as ctrlm
from src.models.plane_model import plane_model
from src.utils.scales import angle_normalize
from src.utils.conversion import deg_to_rad, rad_to_deg

## ONLY LONGITUDINAL CHANNEL
class ControlEnv(gym.Env): 
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, plane_model: plane_model, deltae_max: float, deltaT_max: float, ref_x: list, render_mode: Optional[str] = None, dt: float = 0.5):
        super().__init__()

        self.plane_model = plane_model
        self.ref_x = ref_x
        self.state = np.array([0, 0, 0, 0], dtype=np.float32)

        self.A = self.plane_model.lon['A']
        self.B = self.plane_model.lon['B']

        self.dt = dt
        self.t_hat = 2 * self.plane_model.FC['us'] / self.plane_model.c #TODO add config to select adim or dim variables

        self.max_deltae = deltae_max
        self.max_deltaT = deltaT_max

        self.sys = ctrl.ss(self.A, self.B, np.eye(4), np.zeros((4, 2)))

        self.xs, self.us = [], []
        self.ts = []
        self.ts.append(0)
        self.xs.append(self.state)
        self.us.append(np.array([0, 0], dtype=np.float32)) #TODO add dict to input initial conditionss
        self.last_u = None

        self.render_mode = render_mode

        u_max = np.array([self.max_deltae, self.max_deltaT])
        x_max = np.array([10., 0.75, 1., 1.])

        self.action_space = spaces.Box(
            low=-u_max, high=u_max, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

    def step(self, action):
        du, alpha, q, theta= self.state[0], self.state[1], self.state[2], self.state[3]
        deltae, deltaT = action

        u = np.array([deltae, deltaT], dtype=np.float32)
        x = np.array([du, alpha, q, theta], dtype=np.float32)
        u[0] = np.clip(u[0], -self.max_deltae, self.max_deltae)
        u[1] = np.clip(u[1], -self.max_deltaT, self.max_deltaT)
        t_n = self.ts[-1]

        #reward = -(0.1*angle_normalize(alpha - self.ref_x[1]) ** 2 
        #           + angle_normalize(theta  - self.ref_x[2]) ** 2 
        #           + 0.1*angle_normalize(q  - self.ref_x[3]) ** 2 
        #           + 0.001*(u - self.ref_x[0]))**2

        #reward = 100*(angle_normalize(theta - self.ref_x[3])) ** 2 #TODO add better reward function

        #u = [-0.15708, 0.]

        #print(f"alpha: {alpha}, theta: {theta}, q: {q}, reward: {reward}, @ t = {t_n} s")

        if t_n > 500:
            done = True #TODO add condition for reaching the goal & information display
            self.print_state()
        else:
            done = False

        response = ctrl.forced_response(self.sys, T=[t_n * self.t_hat , (t_n + self.dt) * self.t_hat], U=[u, u], X0=x, squeeze=True)
        x_n = response.outputs[:, 1]

        self.state = x_n

        reward = self.get_reward_function(u)

        self.us.append(u)
        self.xs.append(x_n)
        self.ts.append(t_n + self.dt)

        self.last_u = u

        if self.render_mode == "human":
            self.render()
            self.fig, self.axs = plt.subplots(2)
            self.axs[0].set_title('Observation Variables')
            self.axs[1].set_title('Action Variables') #TODO function that plots nicely the results

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            x_max = np.array([1, 1, 1, 1], dtype=np.float32)
            self.state = np.array([0, 0, 0, 0], dtype=np.float32)
            self.xs, self.us, self.ts = [], [], []
            self.xs.append(self.state)
            self.us.append(np.array([0, 0], dtype=np.float32))
            self.ts.append(0)
            if self.render_mode == "human":
                self.axs[0].clear()
                self.axs[1].clear()

        return (self._get_obs(), {})
    
    def _get_obs(self):
        du, alpha, q, theta = self.state
        return np.array([du, alpha, q, theta], dtype=np.float32)

    def render(self):
        self.axs[0].clear()
        self.axs[1].clear()

        self.axs[0].plot(self.xs)
        self.axs[1].plot(self.us) #TODO function that plots nicely the results
        plt.pause(0.01)

        return None

    def close(self):
        plt.close(self.fig)

    def print_state(self):
        du, alpha, q, theta = self.state
        u = self.last_u
        reward = self.get_reward_function(u)
        print(f"u: {du}, alpha: {rad_to_deg(alpha)}, q: {rad_to_deg(q)}, theta: {rad_to_deg(theta)}")
        print(f"deltae: {rad_to_deg(self.last_u[0])}, deltaT: {self.last_u[1]}")
        print(f"final reward: {reward}")
        print("========================================")
        return None

    def plot(self):
        fig, axs = plt.subplots(4, 1, sharex=True)
        for i, label in enumerate(['u', 'alpha', 'q', 'theta']):
            axs[i].plot(self.ts, [rad_to_deg(xs[i]) for xs in self.xs], label=label)
            axs[i].set_ylabel('Values')
            axs[i].legend()
            axs[i].grid(True)
        axs[0].set_title('Observation Variables')
        axs[-1].set_xlabel('Time')
        plt.tight_layout()

        # Plot the control variables
        fig, axs = plt.subplots(2)
        for i, label in enumerate(['deltae', 'deltaT']):
            axs[i].plot(self.ts, [us[i] for us in self.us], label=label)
            axs[i].set_ylabel('Values')
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()

    def get_reward_function(self, u, info=None):
        x_n = self.state

        # Reward based on the difference between the current state and the reference state
        #state_reward = np.sum((x_n[1:4] - self.ref_x[1:4])**2)
        state_reward = np.sum((x_n[0:3] - self.ref_x[0:3])**2)
        # Reward based on the magnitude of the action taken
        action_reward = np.abs(u[0] / self.max_deltae) + np.abs(u[1] / self.max_deltaT)

            # Reward based on the change in action
        if self.last_u is not None:
            smoothness_reward = -np.sum(np.abs(x_n - self.xs[-1]))
        else:
            smoothness_reward = 0

        # Total reward is a combination of the state reward and the action reward
        reward = 0.6*state_reward + 0.4*action_reward + 0*smoothness_reward

        return -reward