import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import warnings
import logging
import random
import torch
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from aquacropgymnasium.env import Maize

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

seeds = [1, 2, 3]
timestep_values = [500000, 1000000, 1500000, 2000000, 2500000]

train_output_dir = './train_output'
eval_output_dir = './eval_output'
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(eval_output_dir, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_eval_env(seed):
    def _init():
        env = Monitor(Maize(
            mode='eval',
            year1=2008,
            year2=2018
        ))
        set_seed(seed)
        return env
    eval_env = DummyVecEnv([_init])
    return eval_env

def evaluate_agent(agent, env, n_eval_episodes=100, agent_name="Agent"):
    print(f"Starting evaluation for {agent_name}...")
    yields, irrigations = [], []
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            dry_yield = info[0].get('dry_yield', 0)
            total_irrigation = info[0].get('total_irrigation', 0)
        yields.append(dry_yield)
        irrigations.append(total_irrigation)
    return {
        'mean_yield': np.mean(yields),
        'std_yield': np.std(yields),
        'mean_irrigation': np.mean(irrigations),
        'std_irrigation': np.std(irrigations)
    }

def load_ppo_model_timestep(seed, timestep):
    set_seed(seed)
    eval_env = make_eval_env(seed)
    vecnormalize_filename = os.path.join(train_output_dir, f"ppo_model_{timestep}_vecnormalize.pkl")
    eval_env = VecNormalize.load(vecnormalize_filename, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model_name = os.path.join(train_output_dir, f"ppo_model_{timestep}.zip")
    model = PPO.load(model_name, env=eval_env)
    return model, eval_env

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return [action], state

def make_random_env(seed):
    def _init():
        env = Monitor(Maize(
            mode='eval',
            year1=2008,
            year2=2018
        ))
        set_seed(seed)
        return env
    random_env = DummyVecEnv([_init])
    return random_env

def aggregate_results_across_seeds(results_across_seeds):
    mean_yields = [res['mean_yield'] for res in results_across_seeds]
    std_yields = [res['std_yield'] for res in results_across_seeds]
    mean_irrigations = [res['mean_irrigation'] for res in results_across_seeds]
    std_irrigations = [res['std_irrigation'] for res in results_across_seeds]
    return {
        'mean_yield': np.mean(mean_yields),
        'std_yield': np.mean(std_yields),
        'mean_irrigation': np.mean(mean_irrigations),
        'std_irrigation': np.mean(std_irrigations)
    }

CROP_PRICE = 180
IRRIGATION_COST = 1
FIXED_COST = 1728

def calculate_profit(mean_yield, mean_irrigation):
    return CROP_PRICE * mean_yield - IRRIGATION_COST * mean_irrigation - FIXED_COST

def calculate_water_efficiency(mean_yield, mean_irrigation):
    if mean_irrigation == 0:
        return np.nan
    yield_kg_per_ha = mean_yield * 1000
    return yield_kg_per_ha / mean_irrigation

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)
sim_start = "2008/05/01"
sim_end = "2018/12/31"
soil = Soil('SandyLoam')
crop_obj = Crop('Maize', planting_date='05/01')
initWC = InitialWaterContent(value=['FC'])

rainfed = IrrigationManagement(irrigation_method=0)
threshold4_irrigate = IrrigationManagement(irrigation_method=1, SMT=[23.72, 26.46, 38.19, 50.11] * 4)
interval_7 = IrrigationManagement(irrigation_method=2, IrrInterval=7)
net_irrigation = IrrigationManagement(irrigation_method=4, NetIrrSMT=70)
strategies = [rainfed, threshold4_irrigate, interval_7, net_irrigation]
labels = ['Rainfed', 'Thresholds', 'Interval', 'Net']
outputs = []
for i, irr_mngt in enumerate(strategies):
    crop_obj.Name = labels[i]
    model = AquaCropModel(sim_start, sim_end, wdf, soil, crop_obj,
                          initial_water_content=initWC,
                          irrigation_management=irr_mngt)
    model.run_model(till_termination=True)
    outputs.append(model._outputs.final_stats)

outlist = []
for i in range(len(outputs)):
    temp = outputs[i][['Season', 'Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']].copy()
    temp['label'] = labels[i]
    outlist.append(temp)
aquacrop_results = pd.concat(outlist, axis=0)

aquacrop_results_agg = aquacrop_results.groupby('label').agg({
    'Dry yield (tonne/ha)': ['mean', 'std'],
    'Seasonal irrigation (mm)': ['mean', 'std']
}).reset_index()

aquacrop_results_agg.columns = ['label'] + ['_'.join(col).strip() for col in aquacrop_results_agg.columns.values[1:]]

for idx, row in aquacrop_results_agg.iterrows():
    mean_yield = row['Dry yield (tonne/ha)_mean']
    mean_irrigation = row['Seasonal irrigation (mm)_mean']
    aquacrop_results_agg.loc[idx, 'Profit_mean'] = calculate_profit(mean_yield, mean_irrigation)
    aquacrop_results_agg.loc[idx, 'WaterEfficiency_mean'] = calculate_water_efficiency(mean_yield, mean_irrigation)

random_results_across_seeds = []
for seed in seeds:
    random_env = make_random_env(seed)
    random_agent = RandomAgent(random_env.action_space)
    random_results = evaluate_agent(random_agent, random_env, n_eval_episodes=100,
                                    agent_name=f"RandomAgent_seed_{seed}")
    random_results_across_seeds.append(random_results)

random_final_results = aggregate_results_across_seeds(random_results_across_seeds)
random_profit = calculate_profit(random_final_results['mean_yield'], random_final_results['mean_irrigation'])
random_water_efficiency = calculate_water_efficiency(random_final_results['mean_yield'], random_final_results['mean_irrigation'])

random_df = pd.DataFrame({
    'Dry yield (tonne/ha)_mean': [random_final_results['mean_yield']],
    'Dry yield (tonne/ha)_std': [random_final_results['std_yield']],
    'Seasonal irrigation (mm)_mean': [random_final_results['mean_irrigation']],
    'Seasonal irrigation (mm)_std': [random_final_results['std_irrigation']],
    'Profit_mean': [random_profit],
    'WaterEfficiency_mean': [random_water_efficiency],
    'label': ['Random']
})

# Evaluate PPO models at different timesteps
ppo_timesteps_results = pd.DataFrame()
for timestep in timestep_values:
    ppo_results_across_seeds_ts = []
    for seed in seeds:
        try:
            ppo_model, ppo_eval_env = load_ppo_model_timestep(seed, timestep)
            ppo_results = evaluate_agent(ppo_model, ppo_eval_env, n_eval_episodes=100,
                                         agent_name=f"PPO_{timestep}_seed_{seed}")
            ppo_results_across_seeds_ts.append(ppo_results)
        except Exception as e:
            print(f"Could not load model for seed {seed} at timestep {timestep}: {e}")
            continue
    if ppo_results_across_seeds_ts:
        ppo_final_results_ts = aggregate_results_across_seeds(ppo_results_across_seeds_ts)
        ppo_profit_ts = calculate_profit(ppo_final_results_ts['mean_yield'], ppo_final_results_ts['mean_irrigation'])
        ppo_water_efficiency_ts = calculate_water_efficiency(ppo_final_results_ts['mean_yield'], ppo_final_results_ts['mean_irrigation'])
        
        new_row = pd.DataFrame([{
            'Dry yield (tonne/ha)_mean': ppo_final_results_ts['mean_yield'],
            'Dry yield (tonne/ha)_std': ppo_final_results_ts['std_yield'],
            'Seasonal irrigation (mm)_mean': ppo_final_results_ts['mean_irrigation'],
            'Seasonal irrigation (mm)_std': ppo_final_results_ts['std_irrigation'],
            'Profit_mean': ppo_profit_ts,
            'WaterEfficiency_mean': ppo_water_efficiency_ts,
            'label': f'PPO_{timestep}'
        }])
        
        # Concatenate the new row to the existing DataFrame
        ppo_timesteps_results = pd.concat([ppo_timesteps_results, new_row], ignore_index=True)

ppo_timesteps_results.to_csv(os.path.join(eval_output_dir, 'ppo_timesteps_results.csv'), index=False)

best_ppo = ppo_timesteps_results.loc[ppo_timesteps_results['Profit_mean'].astype(float).idxmax()].copy()
best_ppo['label'] = 'PPO'

comparison_df = pd.concat([random_df, aquacrop_results_agg, best_ppo.to_frame().T], ignore_index=True)
comparison_df.to_csv(os.path.join(eval_output_dir, 'comparison_results.csv'), index=False)

desired_order = ['PPO', 'Thresholds', 'Interval', 'Net', 'Rainfed', 'Random']
comparison_df['label'] = pd.Categorical(comparison_df['label'], categories=desired_order, ordered=True)
comparison_df = comparison_df.sort_values('label')

combined_labels = comparison_df['label']
combined_yields = comparison_df['Dry yield (tonne/ha)_mean'].astype(float)
combined_irrigations = comparison_df['Seasonal irrigation (mm)_mean'].astype(float)
combined_profits = comparison_df['Profit_mean'].astype(float)
combined_water_efficiency = comparison_df['WaterEfficiency_mean'].astype(float)
combined_yields_std = comparison_df['Dry yield (tonne/ha)_std'].astype(float)
combined_irrigations_std = comparison_df['Seasonal irrigation (mm)_std'].astype(float)

color_dict = {
    'PPO': '#0072B2',          # Blue
    'Thresholds': '#009E73',   # Green
    'Interval': '#D55E00',     # Orange
    'Net': '#CC79A7',          # Purple
    'Rainfed': '#F0E442',      # Yellow
    'Random': '#999999'        # Grey
}
colors = [color_dict[label] for label in combined_labels]

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_yields, yerr=combined_yields_std, capsize=5, color=colors, edgecolor='black')
plt.ylabel('Mean yield (tonne/ha)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_yields.png'), format='png', dpi=300)
plt.close()

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_irrigations, yerr=combined_irrigations_std, capsize=5, color=colors, edgecolor='black')
plt.ylabel('Total irrigation (mm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_irrigations.png'), format='png', dpi=300)
plt.close()

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_profits, color=colors, edgecolor='black')
plt.ylabel('Profit ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_profits.png'), format='png', dpi=300)
plt.close()

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_water_efficiency, color=colors, edgecolor='black')
plt.ylabel('Water efficiency (kg/m³)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_water_efficiency.png'), format='png', dpi=300)
plt.close()

print("Evaluation completed.")