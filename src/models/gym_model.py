from typing import Optional
import pprint as pp

import numpy as np

from gymnasium import spaces
import gymnasium as gym

import control as ctrl
from src.models.plane_model import plane_model
from src.utils.conversion import rad_to_deg
from src.utils.scales import get_abs_error, get_rel_error, denormalize
from src.utils.scales import change_reference_rand_interval

## GLOBAL VARIABLES

SETTLED_REWARD = 2

THETA_TRAIN = 2 # Deg
PSI_TRAIN = 5 # Deg

T_MIN_TRAIN = 250
T_MAX_TRAIN = 500

## ------------------------------- LONGITUDINAL DIRECTIONAL CHANNEL -------------------------------------
class ControlEnv(gym.Env): 
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, plane_model: plane_model, deltae_max: float, deltaT_max: float, ref_x: list, render_mode: Optional[str] = None, dt: float = 0.5, tmax = 500, ss_tick_max = 30, X0 = [0., 0., 0., 0.], train_mode = False, muT = 0.5):
        super().__init__()

        self.plane_model = plane_model

        if train_mode:
            self.ref_vector = [[], [], []] # TODO lat objectives
            self.ref_vector[2] = change_reference_rand_interval(dt, np.deg2rad(-THETA_TRAIN), np.deg2rad(THETA_TRAIN), tmax, T_MIN_TRAIN, T_MAX_TRAIN) #theta only TODO include more options
        else:
            self.ref_vector = [[], [], []]
            self.ref_vector[2] = [ref_x[2] for i in range(int(tmax/dt))]

        self.ref_x = [0, 0, self.ref_vector[2][0]] #[beta, phi, theta]

        X0.append(self.ref_x[2])
        X0.append(get_abs_error(0, ref_x[2]) ** 2)
        X0.extend([X0[3], X0[3], X0[3], X0[3], X0[3]]) #theta_n-5, theta_n-4, theta_n-3, theta_n-2, theta_n-1

        self.dt = dt
        self.t_hat = self.plane_model.lon['t_lon'] #TODO add config to select adim or dim variables
        self.tmax = tmax
        self.mu_T = muT

        self.max_deltae = deltae_max
        self.max_deltaT = deltaT_max

        self.ss_counter = 0
        self.ss_tick_max = ss_tick_max

        self.sum_rewards = 0
        self.reward_x = []

        self.sys = ctrl.ss(plane_model.lon['A'], plane_model.lon['B'], np.eye(4), np.zeros((4, 2)))

        self.xs, self.us = [], []
        self.ts = []
        self.ts.append(0)
        self.xs.append(X0)
        self.us.append(np.array([0, 0], dtype=np.float32))
        self.last_u = None

        self.render_mode = render_mode
        self.train_mode = train_mode

        # Define the minimum and maximum values for each observation and action variable
        self.obs_max = np.array([10., np.pi, np.pi, np.pi, np.pi, 1., np.pi, np.pi, np.pi, np.pi, np.pi])
        self.obs_min = -1 * np.array([10., np.pi, np.pi, np.pi, np.pi, 0., np.pi, np.pi, np.pi, np.pi, np.pi])
                                #du, alpha, q, theta, theta_ref, e(t)^2, theta_n-5, theta_n-4, theta_n-3, theta_n-2, theta_n-1
        self.action_min = np.array([-self.max_deltae, -self.max_deltaT])
        self.action_max = np.array([self.max_deltae, self.max_deltaT])

        u_max = np.array([1, 0.5])
        x_max = np.array([10., 0.5, 0.5, 0.5, 0.5, 1., 0.5, 0.5, 0.5, 0.5, 0.5])
                        #du, alpha, q, theta, theta_ref, e(t)^2, theta_n-4, theta_n-4, theta_n-3, theta_n-2, theta_n-1

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, 1]), shape=(2,), dtype=np.float32
        )

                # Calculate the maximum possible change in action
        self.max_action_change = np.abs(self.action_space.high - self.action_space.low)

        self.observation_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        self.state = np.array(X0, dtype=np.float32)

    def step(self, action):
        du, alpha, q, theta= self.state[0], self.state[1], self.state[2], self.state[3]

        action[0] = np.clip(action[0], 0, 1)
        action[1] = np.clip(action[1], 0, 1)

        deltae = denormalize(action[0], -self.max_deltae, self.max_deltae)
        deltaT = denormalize(action[1], -self.max_deltaT, self.max_deltaT)

        u = np.array([deltae, deltaT], dtype=np.float32)
        x = np.array([du, alpha, q, theta], dtype=np.float32)

        t_n = self.ts[-1]

        response = ctrl.forced_response(self.sys, T=[t_n / self.t_hat , (t_n + self.dt) / self.t_hat], U=[self.us[-1], u], X0=x, squeeze=True, interpolate=True)
        x_n = response.outputs[:, 1]
        self.state[0:4] = x_n

        reward = self.get_reward_function(u)

        # Update the reference vector
        self.ref_x[2] = self.ref_vector[2][len(self.ts) - 1]
        self.state[4] = self.ref_x[2]

        # Compute theta error squared
        self.state[5] = (self.state[3] - self.ref_x[2]) ** 2

        # Shift the last 5 theta readings to the left
        self.state[6:10] = self.state[7:11]

        # Store the current state as the last state
        self.state[10] = self.state[3]

        self.last_u = u

        terminated = False
        truncated = False

        # Store the current action and state
        self.us.append(u)
        self.xs.append(x_n)
        self.ts.append(t_n + self.dt)

        if self.is_settled():
            reward += SETTLED_REWARD

        ## Check if the system has reached the time limit
        if t_n > self.tmax:
            terminated = True
            truncated = True
            pp.pprint(self._get_info())

        self.sum_rewards += reward

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)
        if options is None:
            if self.train_mode:
                self.ref_vector = [[], [], []]
                self.ref_vector[2] = change_reference_rand_interval(self.dt, np.deg2rad(-2), np.deg2rad(2), self.tmax, 250, 500) #theta only
                self.ref_x[2] = self.ref_vector[2][0]

            self.state = np.array([0, 0, 0, 0, self.ref_x[2],get_abs_error(0, self.ref_x[2]) ** 2, 0, 0, 0, 0, 0], dtype=np.float32) # TODO include X0
            self.xs, self.us, self.ts = [], [], []
            self.xs.append(np.array([0, 0, 0, 0], dtype=np.float32))
            self.reward_x.append(self.sum_rewards)
            self.sum_rewards = 0
            self.last_u = None
            self.previous_action = np.array([0.5, 0.5])
            self.us.append(np.array([0, 0], dtype=np.float32))
            self.ts.append(0)

        return (self._get_obs(), {})
    
    def _get_obs(self):
        du, alpha, q, theta, theta_ref, e2, theta_n5, theta_n4, theta_n3, theta_n2, theta_n1 = self.state
        return np.array([du, alpha, q, theta, theta_ref, e2, theta_n5, theta_n4, theta_n3, theta_n2, theta_n1], dtype=np.float32)

    def render(self):

        return None

    def close(self):

        return None
    
    def _get_info(self):
        u = self.last_u if self.last_u is not None else [0, 0]
        info = {
            "objective [beta, phi, theta]": np.rad2deg(self.ref_x),
            "state lon [u, alpha, q, theta]": [self.state[0]*self.plane_model.FC['us'], self.state[1]*180/np.pi, self.state[2]*180/np.pi*(self.plane_model.FC['us']*2/self.plane_model.c), self.state[3]*180/np.pi],
            "state lat []": 0,
            "action lon [deltae, deltaT]": [rad_to_deg(u[0]), u[1]],
            "action lat [deltaa, deltar]": "None",
            "action diff [deltae, deltaT]": [np.mean(np.square(np.diff(np.array(self.us)[:, 0]))), np.mean(np.square(np.diff(np.array(self.us)[:, 1])))],
            "time": self.ts[-1],
            "stationary error lon (%)": get_rel_error(self.state[3], self.ref_x[2])*100,
            "last_reward": self.get_reward_function(u),
            "sum_rewards": self.sum_rewards,
            "is_settled": self.is_settled(),
            "is_out_of_bounds": self.is_out_of_bounds()
        }
        return info

    def get_reward_function(self, u, join = True):
        theta = self.state[3]
        theta_ref = self.ref_x[2]

        e_t = (theta_ref - theta) / theta_ref

        # Use a smoother function for error scaling
        scaled_error = np.tanh(np.abs(e_t))

        #r_state = np.exp(-(e_t/self.mu_T) ** 2)

        cost_penalty = 0.1 * (np.abs(u[0] / self.max_deltae) ** 2 + np.abs(u[1] / self.max_deltaT) ** 2)

        # Action change penalty (normalized by the range of action space) (OLD, for reference only)
        # action_change_penalty = (np.abs(u[0] - self.us[0][-2]) / self.max_deltae) ** 2 + (np.abs(u[1] - self.us[1][-2]) / self.max_deltaT) ** 2 if self.last_u is not None else 0

        return 1 - 2 * scaled_error - cost_penalty
    
    def is_settled(self, threshold=0.02):
        """
        Check if the system's output y has reached the settling time.

        Parameters:
        threshold: Percentage of the final value that defines the settling time. Default is 0.02 (2%).

        Returns:
        True if the system's output has reached the settling time, False otherwise.
        """
        theta = self.state[3]
        theta_ref = self.ref_x[2]
        if np.abs((theta_ref - theta) /  theta_ref) < threshold:
            self.ss_counter += 1
            if self.ss_counter >= self.ss_tick_max:
                return True
            return False
        # if the difference is not lie around 2% of reference consecutively,
        # reset the counter
        self.ss_counter = 0
        return False
    
    def is_out_of_bounds(self):
        theta = self.state[3]

        if np.abs(theta) > (10 * np.pi) / 180:
            return True

        return False

## ------------------------------- LATERAL DIRECTIONAL CHANNEL -------------------------------------
class ControlEnv_Lat(gym.Env): 
    """Custom Environment that follows gym interface""" # TODO: Add proper descriptions

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, plane_model: plane_model, deltaa_max: float, deltar_max: float, ref_x: list, render_mode: Optional[str] = None, dt: float = 0.5, tmax = 500, ss_tick_max = 30, X0 = [0., 0., 0., 0., 0.], train_mode = False, muT = 0.5, lambda_cost = 0.25, lambda_lat = 5):
        super().__init__()

        self.plane_model = plane_model

        if train_mode:
            self.ref_vector = [[], [], []] # phi psi theta
            self.ref_vector[0] = [0 for i in range(int(tmax/dt) + 2)]
            self.ref_vector[1] = change_reference_rand_interval(dt, np.deg2rad(-PSI_TRAIN), np.deg2rad(PSI_TRAIN), tmax, T_MIN_TRAIN, T_MAX_TRAIN)

        else:
            self.ref_vector = [[], [], []] # phi psi theta
            self.ref_vector[0] = [ref_x[0] for i in range(int(tmax/dt))]
            self.ref_vector[1] = [ref_x[1] for i in range(int(tmax/dt))]

        self.ref_x = [self.ref_vector[0][0], self.ref_vector[1][0], 0] #[phi, psi, theta]

        X0.append(self.ref_x[0])
        X0.append(self.ref_x[1])

        X0.append(get_abs_error(0, ref_x[0]) ** 2)
        X0.append(get_abs_error(0, ref_x[1]) ** 2)

        #X0.extend([X0[3], X0[3], X0[3], X0[3], X0[3]]) #theta_n-5, theta_n-4, theta_n-3, theta_n-2, theta_n-1

        self.lambda_cost = lambda_cost
        self.lambda_lat = lambda_lat

        self.dt = dt
        self.t_hat = self.plane_model.lat['t_lat'] #TODO add config to select adim or dim variables
        self.tmax = tmax
        self.mu_T = muT

        self.max_deltaa = deltaa_max
        self.max_deltar = deltar_max

        self.ss_counter = 0
        self.ss_tick_max = ss_tick_max

        self.sum_rewards = 0
        self.reward_x = []

        # Parámetros del filtro de paso bajo
        omega = 10.0  # Frecuencia de corte (rad/s)

        # Factor de atenuación del filtro
        alpha = omega * (self.dt  / self.t_hat ) / (1 + omega * (self.dt  / self.t_hat))

        self.sys = ctrl.ss(plane_model.lat['A'], plane_model.lat['B'], np.eye(5), np.zeros((5, 2)))

        self.xs, self.us = [], []
        self.ts = []
        self.ts.append(0)
        self.xs.append(X0)
        self.us.append(np.array([0, 0], dtype=np.float32))
        self.last_u = None

        self.render_mode = render_mode
        self.train_mode = train_mode

        # Define the minimum and maximum values for each observation and action variable
        #self.obs_max = np.array([10., np.pi, np.pi, np.pi, np.pi, 1., np.pi, np.pi, np.pi, np.pi, np.pi])
        #self.obs_min = -1 * np.array([10., np.pi, np.pi, np.pi, np.pi, 0., np.pi, np.pi, np.pi, np.pi, np.pi])
        #                        #du, alpha, q, theta, theta_ref, e(t)^2, theta_n-5, theta_n-4, theta_n-3, theta_n-2, theta_n-1
        #self.action_min = np.array([-self.max_deltae, -self.max_deltaT])
        #self.action_max = np.array([self.max_deltae, self.max_deltaT])

        u_max = np.array([1, 0.5])
        x_max = np.array([50., 50., 50., 50., 50., 50., 50., 50., 50.])
                        #beta, p, r, phi, psi, phi_ref, psi_ref, e_phi^2, e_psi^2

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, 1]), shape=(2,), dtype=np.float32
        )

        # Calculate the maximum possible change in action
        self.max_action_change = np.abs(self.action_space.high - self.action_space.low)

        self.observation_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        self.state = np.array(X0, dtype=np.float32)

    def step(self, action):
        beta, p, r, phi, psi= self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]

        action[0] = np.clip(action[0], 0, 1)
        action[1] = np.clip(action[1], 0, 1)

        #action[0] = lfilter(b, a, [action[0]])[-1]
        #action[1] = lfilter(b, a, [action[1]])[-1]

        deltaa = denormalize(action[0], -self.max_deltaa, self.max_deltaa)
        deltar = denormalize(action[1], -self.max_deltar, self.max_deltar)

        u = np.array([deltaa, deltar], dtype=np.float32)
        x = np.array([beta, p, r, phi, psi], dtype=np.float32)

        t_n = self.ts[-1]

        response = ctrl.forced_response(self.sys, T=[t_n / self.t_hat , (t_n + self.dt) / self.t_hat], U=[self.us[-1], u], X0=x, squeeze=True, interpolate=True)
        x_n = response.outputs[:, 1]
        self.state[0:5] = x_n

        reward = self.get_reward_function(u)

        # Update the reference vector
        self.ref_x[0] = self.ref_vector[0][len(self.ts) - 1]
        self.ref_x[1] = self.ref_vector[1][len(self.ts) - 1]

        self.state[5] = self.ref_x[0]
        self.state[6] = self.ref_x[1]

        # Compute theta error squared
        self.state[7] = (self.state[3] - self.ref_x[0]) ** 2
        self.state[8] = (self.state[4] - self.ref_x[1]) ** 2

        self.last_u = u

        terminated = False
        truncated = False

        # Store the current action and state
        self.us.append(u)
        self.xs.append(x_n)
        self.ts.append(t_n + self.dt)

        if self.is_settled():
            reward += SETTLED_REWARD

        ## Check if the system has reached the time limit
        if t_n >= self.tmax:
            terminated = True
            truncated = True
            pp.pprint(self._get_info())

        self.sum_rewards += reward

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)
        if options is None:
            if self.train_mode:
                self.ref_vector = [[], [], []]
                self.ref_vector[0] = [0 for i in range(int(self.tmax/self.dt) + 2)]
                self.ref_vector[1] = change_reference_rand_interval(self.dt, np.deg2rad(-5), np.deg2rad(5), self.tmax, 250, 500)
                self.ref_x[0] = self.ref_vector[0][0]
                self.ref_x[1] = self.ref_vector[1][0]

            self.state = np.array([0, 0, 0, 0, 0, self.ref_x[0], self.ref_x[1], get_abs_error(0, self.ref_x[0]) ** 2, get_abs_error(0, self.ref_x[1]) ** 2], dtype=np.float32) # TODO include X0
            self.xs, self.us, self.ts = [], [], []
            self.xs.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
            self.reward_x.append(self.sum_rewards)
            self.sum_rewards = 0
            self.last_u = None
            self.previous_action = np.array([0.5, 0.5])
            self.us.append(np.array([0, 0], dtype=np.float32))
            self.ts.append(0)

        return (self._get_obs(), {})
    
    def _get_obs(self):
        beta, p, r, phi, psi, phi_ref, psi_ref, e2_phi, e2_psi = self.state
        return np.array([beta, p, r, phi, psi, phi_ref, psi_ref, e2_phi, e2_psi], dtype=np.float32)

    def render(self):

        return None

    def close(self):

        return None
    
    def _get_info(self): 
        u = self.last_u
        info = {
            "objective [beta, phi, theta]": np.rad2deg(self.ref_x),
            "state lon [u, alpha, q, theta]": 0,
            "state lat [beta, p, r, phi, psi]": [self.state[0]*180/np.pi, self.state[1]*180/np.pi, self.state[2]*180/np.pi, self.state[3]*180/np.pi, self.state[4]*180/np.pi],
            "action lat [deltaa, deltar]": [rad_to_deg(u[0]), rad_to_deg(u[1])],
            "action diff [deltaa, deltar]": [np.mean(np.square(np.diff(np.array(self.us)[:, 0]))), np.mean(np.square(np.diff(np.array(self.us)[:, 1])))],
            "time": self.ts[-1],
            "stationary error lat (%)": [get_rel_error(self.state[3], self.ref_x[0])*100, get_rel_error(self.state[4], self.ref_x[1])*100],
            "last_reward": self.get_reward_function(u),
            "sum_rewards": self.sum_rewards,
            "is_settled": self.is_settled()
        }
        return info

    def get_reward_function(self, u, join = True): 
        beta, p, r, phi, psi= self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]

        psi_ref = self.ref_x[1]

        e_t_psi = np.abs((psi_ref - psi) / psi_ref)

        cost_penalty = self.lambda_cost * (np.abs(u[0] / self.max_deltaa) ** 2 + np.abs(u[1] / self.max_deltar) ** 2)

        # Action change penalty (normalized by the range of action space)
        #action_change_penalty = (np.abs(u[0] - self.us[0][-2]) / self.max_deltaa) ** 2 + (np.abs(u[1] - self.us[1][-2]) / self.max_deltar) ** 2 if self.last_u is not None else 0

        return 1 - np.tanh(e_t_psi) - self.lambda_lat*(np.abs(p) + np.abs(beta) + np.abs(phi)) - cost_penalty
    
    def is_settled(self, threshold=0.02):
        """
        Check if the system's output y has reached the settling time.

        Parameters:
        threshold: Percentage of the final value that defines the settling time. Default is 0.02 (2%).

        Returns:
        True if the system's output has reached the settling time, False otherwise.
        """
        psi = self.state[4]

        psi_ref = self.ref_x[1]

        if np.abs((psi_ref - psi) /  psi_ref) < threshold:
            self.ss_counter += 1
            if self.ss_counter >= self.ss_tick_max:
                return True
            return False
        # if the difference is not lie around 2% of reference consecutively,
        # reset the counter
        self.ss_counter = 0
        return False
