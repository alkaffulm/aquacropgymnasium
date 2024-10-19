import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import warnings
import logging

from aquacropgymnasium.env import Maize

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

output_dir = './train_output'
os.makedirs(output_dir, exist_ok=True)

class RewardLoggingCallback(BaseCallback):
    def __init__(self, agent_name, output_dir, verbose=0, num_intervals=100):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.agent_name = agent_name
        self.output_dir = output_dir
        self.num_intervals = num_intervals
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
        rewards = np.array(self.episode_rewards)
        if len(rewards) >= self.num_intervals:
            interval_size = len(rewards) // self.num_intervals
            avg_rewards = np.array([
                np.mean(rewards[i * interval_size:(i + 1) * interval_size])
                for i in range(self.num_intervals)
            ])
            x = np.arange(self.num_intervals) * interval_size
            plt.figure(figsize=(10, 5))
            plt.plot(x, avg_rewards, marker='o')
            plt.xlabel('Episodes')
            plt.ylabel('Average Total Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f'reward_plot_{self.agent_name}.png'), format='png', dpi=300)
            plt.close()
        else:
            x = np.arange(len(rewards))
            plt.figure(figsize=(10, 5))
            plt.plot(x, rewards, marker='o')
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f'reward_plot_{self.agent_name}.png'), format='png', dpi=300)
            plt.close()

timestep_values = [500000, 1000000, 1500000, 2000000, 2500000]

for train_timesteps in timestep_values:
    env = Maize(mode='train', year1=1982, year2=2007)
    train_env = DummyVecEnv([lambda: Monitor(env)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model_name = f"ppo_model_{train_timesteps}"
    log_dir = os.path.join(output_dir, f"tensorboard_logs_{train_timesteps}")
    os.makedirs(log_dir, exist_ok=True)

    ppo_reward_logging_callback = RewardLoggingCallback(
        agent_name=model_name,
        output_dir=output_dir,
        num_intervals=100
    )

    ppo_model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=6.34e-04,
        n_steps=2048,
        batch_size=512,
        n_epochs=23,
        gamma=0.98,
        clip_range=0.22,
        ent_coef=4.50e-04,
        verbose=1,
        tensorboard_log=log_dir
    )

    print(f"Training {model_name} for {train_timesteps} timesteps...")
    ppo_model.learn(total_timesteps=train_timesteps, callback=ppo_reward_logging_callback)

    model_output_path = os.path.join(output_dir, f"{model_name}.zip")
    vecnorm_output_path = os.path.join(output_dir, f"{model_name}_vecnormalize.pkl")

    ppo_model.save(model_output_path)
    train_env.save(vecnorm_output_path)
    train_env.close()

    print(f"Training completed for {model_name}.\n")