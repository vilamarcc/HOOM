from unicodedata import normalize

from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from src.models.gym_model import ControlEnv, ControlEnv_Lat
from src.models import plane_model as model
from data import plane_data
from stable_baselines3.common.env_checker import check_env
import numpy as np

import matplotlib.pyplot as plt
import control as ctrl

from src.models.gym_wrappers import HistoryWrapper, ActionSmoothingWrapper
from src.utils.scales import get_rel_error, denormalize

import numpy as np
import pprint as pp

# Create a GlobalHawk plane object
GH = plane_data.GlobalHawk()

dt = 0.5
tend = 500
reference_values = [0, 0, 0]

# Create a plane model object and load the plane data
plane = model.plane_model()
plane.loadplane(GH)

env = ControlEnv(plane, 5.*np.pi/180, 5.*np.pi/180, reference_values, render_mode= "none", dt = dt, train_mode=True, muT= 0.5, tmax= tend, lambda_lat = 2)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
action_noise_td3 = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

env = ActionSmoothingWrapper(env, smoothing_coef=0.8)
env = HistoryWrapper(env=env)


model_ddpg = DDPG(
    policy = 'MlpPolicy',
    env = env,
    verbose= 1,
    learning_starts = 1,
    tau = 0.001,
    gamma = 0.99,
    train_freq = 500,
    batch_size = 1024,
    buffer_size = 500,
    learning_rate = 0.0012)
 
# Definir la arquitectura de la red neuronal personalizada
policy_kwargs = dict(
    net_arch=[256, 256]  # Aumentar la complejidad de la red para capturar mejor las dinámicas
)

model_sac = SAC(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    learning_rate=0.0012,
    buffer_size=1000000,
    learning_starts=2000,
    batch_size=8,
    tau=0.005,
    gamma=0.90,
    action_noise=action_noise,
    train_freq=250,
    gradient_steps=100,
    use_sde=True,                # Enable SDE
    sde_sample_freq=4            # Frequency of resampling the noise
)

model_td3 = TD3(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    buffer_size=1000000,
    learning_starts=2000,
    batch_size=8,
    tau=0.005,
    train_freq=250,
    gradient_steps=100,
    gamma=0.95,
    learning_rate = 0.0012,
    action_noise=action_noise)          # Frequency of resampling the noise

policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
)

model_ppo = PPO(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    learning_rate=0.0012,          # Disminuir la tasa de aprendizaje para una convergencia más estable
    batch_size=64,                 # Incrementar el tamaño del lote
    gamma=0.95,                    # Incrementar el coeficiente de descuento
    gae_lambda=0.90,               # Ajustar el coeficiente de ventaja generalizada
    clip_range=0.2,
    ent_coef=0.01,                 # Agregar un coeficiente de entropía para mejorar la exploración
    use_sde=True,                  # Habilitar SDE (Exploración Estocástica Diferenciable)
    sde_sample_freq=4               # Frecuencia de resampling del ruido                                              # Limitar la divergencia KL para estabilidad
)

model_sac.set_env(env)
model_ppo.set_env(env)
model_td3.set_env(env)

model_ppo.learn(total_timesteps=2000000, log_interval=10)
model_ppo.save(f"ppo_controller_test_con_smooth_control_lat.zip")
np.save('ppo_reward_data_lat.npy', env.reward_x)
env.reset()

model_td3.learn(total_timesteps=2000000, log_interval=10)
model_td3.save(f"td3_controller_test_con_smooth_control_lat.zip")
np.save('td3_reward_data_lat.npy', env.reward_x)
env.reset()

model_sac.learn(total_timesteps=1700000, log_interval=10)
model_sac.save(f"sac_controller_test_con_smooth_control_lat.zip")
np.save('sac_reward_data_lat.npy', env.reward_x)
env.reset()


## ---------------- Evaluate the model and plot some result plots -------------------------

# Load the trained model

#model = SAC.load("sac_controller_test_con_smooth_control.zip")
#model = PPO.load("ppo_controller_test_con_smooth_control.zip")
model = TD3.load("td3_controller_test_con_smooth_control.zip")

vec_env = env
# Use the trained model to control your system
observations = []
actions = []
ref_theta = []

#obs, _ = vec_env.reset()
obs = [0, 0, 0, 0, reference_values[2], get_rel_error(reference_values[2], 0) **2 , 0, 0, 0, 0, 0]
u = [normalize(np.deg2rad(1), -5.*np.pi/180, 5.*np.pi/180), normalize(0, -0.5, 0.5)]
for _ in range(int(tend/dt)):
    action, _ = model.predict(obs)
    obs, reward, done, info, _ = vec_env.step(action)
    observations.append(obs[0:4])
    ref_theta.append(obs[4])
    actions.append([denormalize(action[0], -5., 5.), denormalize(action[1],-0.5, 0.5)])
    if done:
        break
        obs = vec_env.reset()

pp.pprint(vec_env._get_info())

observations = np.array(observations)
actions = np.array(actions)
ref_theta = np.array(ref_theta)
t = np.linspace(0, tend, len(observations[:,0]))
U = np.deg2rad(1)
# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2)
scales = [plane.FC['us']/(180/np.pi),180/np.pi,180/np.pi*(plane.FC['us']*2/plane.c),180/np.pi]
# Plot each observation variable in a separate subplot
labels = ['du', 'alpha', 'q', 'theta']
for i in range(2):
    for j in range(2):
        index = i * 2 + j
        axs[i, j].plot(t, observations[:,index]*scales[index], label=labels[index])
        axs[i, j].set_title(labels[index])
        axs[i, j].set_ylabel('Observations')
        axs[i, j].grid(True)
        axs[i, j].legend()
        if labels[index] == 'theta':
            axs[i, j].plot(t, ref_theta*scales[index], label='theta_ref')
            axs[i, j].legend()

axs[-1, -1].set_xlabel('Time Step')
plt.tight_layout()

# Create a figure with 2 subplots
fig, axs = plt.subplots(2)
scales = [180/np.pi,1]
# Plot each observation variable in a separate subplot
for i, label in enumerate(['deltae','deltaT']):
    #axs[i].plot(actions[:,0,i]*scales[i], label=label)
    axs[i].plot(t, actions[:,i], label=label)
    axs[i].set_title(label)
    axs[i].set_ylabel('Actions')
    axs[i].grid(True)
    axs[i].legend()
axs[-1].set_xlabel('Time Step')
# Adjust the layout and show the plot
plt.tight_layout()
sys = ctrl.ss(plane.lon['A'], plane.lon['B'], np.eye(4), np.zeros((4, 2)))
# Compute the step response
response = ctrl.step_response(sys, squeeze = True, input=0)

yout = response.outputs
yout *= U
T = response.time*plane.lon['t_lon']
# Create a new figure for the step response
fig, axs = plt.subplots(2, 2)
scales = [plane.FC['us'],180/np.pi,180/np.pi*(plane.FC['us']*2/plane.c),180/np.pi]
# Plot the step response in a separate subplot
labels = ['du', 'alpha', 'q', 'theta']
for i in range(2):
    for j in range(2):
        index = i * 2 + j
        axs[i, j].plot(T, yout[index]*scales[index], label=labels[index])
        axs[i, j].set_ylabel('Response')
        axs[i, j].grid(True)
        axs[i, j].legend()
axs[-1, -1].set_xlabel('Time (s)')
plt.tight_layout()

Gu = plane.lon['G']['Gudeltae']['Gfact']
Galpha = plane.lon['G']['Galphadeltae']['Gfact']
Gq = plane.lon['G']['Gqdeltae']['Gfact']
Gtheta = plane.lon['G']['Gthetadeltae']['Gfact']

# Create a new figure for the step responses
fig, axs = plt.subplots(2, 2)

# List of transfer functions
transfer_functions = [Gu, Galpha, Gq, Gtheta]

# Plot the step response of each transfer function in a separate subplot
for i in range(2):
    for j in range(2):
        index = i * 2 + j
        # Compute the step response
        T, yout = ctrl.step_response(transfer_functions[index])
        yout *= U
        # Plot the step response
        axs[i, j].plot(T, yout*scales[index], label=labels[index])
        axs[i, j].set_ylabel('Response')
        axs[i, j].grid(True)
        axs[i, j].legend()
axs[-1, -1].set_xlabel('Time (s)')
plt.tight_layout()

plt.show()