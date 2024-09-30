import os
import pickle as pkl
import random
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from aquacropgymnasium.env import Maize

warnings.filterwarnings("ignore", category=FutureWarning)

study_path = "optuna_study"

class RewardLoggingCallback(BaseCallback):
    def __init__(self, agent_name, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.agent_name = agent_name
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.total_steps = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.current_episode_rewards.append(reward)
        self.total_steps += 1

        if 'dones' in self.locals and any(self.locals['dones']):
            total_reward = np.sum(self.current_episode_rewards)
            self.episode_rewards.append(total_reward)
            self.current_episode_rewards = []
        return True

    def _on_training_end(self):
        if self.episode_rewards:
            final_mean_reward = np.mean(self.episode_rewards)
            final_std_reward = np.std(self.episode_rewards)
            print(f"Training finished for {self.agent_name}. Final mean reward: {final_mean_reward}, Final reward std: {final_std_reward}")
        self.plot_rewards()

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title(f'Total Reward per Episode - {self.agent_name}')
        plt.grid(True)
        plt.savefig(f'reward_plot_{self.agent_name}.png')
        plt.show()

def optimize_agent(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [2048, 4096, 8192])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    n_epochs = trial.suggest_int('n_epochs', 5, 30)
    gamma = trial.suggest_loguniform('gamma', 0.95, 0.999)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=0
    )
    
    callback = RewardLoggingCallback(agent_name='PPO')

    model.learn(total_timesteps=100000, callback=callback)

    rewards = callback.episode_rewards
    mean_reward = np.mean(rewards[-10:])
    return mean_reward

train_env = DummyVecEnv([lambda: Monitor(Maize(mode='train', year1=1982, year2=2007))])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

ppo_study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
ppo_study.optimize(optimize_agent, n_trials=30)

print("Best PPO hyperparameters:", ppo_study.best_params)

os.makedirs(study_path, exist_ok=True)
with open(f"{study_path}/ppo_study.pkl", "wb+") as f:
    pkl.dump(ppo_study, f)

try:
    fig1 = plot_optimization_history(ppo_study)
    fig2 = plot_param_importances(ppo_study)
    fig3 = plot_parallel_coordinate(ppo_study)
    
    fig1.savefig(f'{study_path}/ppo_optimization_history.png')
    fig2.savefig(f'{study_path}/ppo_param_importance.png')
    fig3.savefig(f'{study_path}/ppo_parallel_coordinate.png')

    fig1.show()
    fig2.show()
    fig3.show()

except (ValueError, ImportError, RuntimeError) as e:
    print("Error during plotting:", e)

ppo_study.trials_dataframe().to_csv(f"{study_path}/ppo_report.csv")
