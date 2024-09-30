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

def load_ppo_model(seed):
    set_seed(seed)
    eval_env = make_eval_env(seed)
    vecnormalize_filename = os.path.join(train_output_dir, "ppo_model_vecnormalize.pkl")
    eval_env = VecNormalize.load(vecnormalize_filename, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model_name = os.path.join(train_output_dir, "ppo_model.zip")
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
    irrigation_m3_per_ha = mean_irrigation * 10
    if irrigation_m3_per_ha == 0:
        return np.nan
    yield_kg_per_ha = mean_yield * 1000
    return yield_kg_per_ha / irrigation_m3_per_ha

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
})

aquacrop_results_agg.columns = ['_'.join(col).strip() for col in aquacrop_results_agg.columns.values]

for label in aquacrop_results_agg.index:
    mean_yield = aquacrop_results_agg.loc[label, 'Dry yield (tonne/ha)_mean']
    mean_irrigation = aquacrop_results_agg.loc[label, 'Seasonal irrigation (mm)_mean']
    aquacrop_results_agg.loc[label, 'Profit_mean'] = calculate_profit(mean_yield, mean_irrigation)
    aquacrop_results_agg.loc[label, 'WaterEfficiency_mean'] = calculate_water_efficiency(mean_yield, mean_irrigation)

ppo_results_across_seeds = []
for seed in seeds:
    try:
        ppo_model, ppo_eval_env = load_ppo_model(seed)
        ppo_results = evaluate_agent(ppo_model, ppo_eval_env, n_eval_episodes=100,
                                     agent_name=f"PPO_seed_{seed}")
        ppo_results_across_seeds.append(ppo_results)
    except Exception as e:
        print(f"Could not load model for seed {seed}: {e}")
        continue

random_results_across_seeds = []
for seed in seeds:
    random_env = make_random_env(seed)
    random_agent = RandomAgent(random_env.action_space)
    random_results = evaluate_agent(random_agent, random_env, n_eval_episodes=100,
                                    agent_name=f"RandomAgent_seed_{seed}")
    random_results_across_seeds.append(random_results)

ppo_final_results = aggregate_results_across_seeds(ppo_results_across_seeds)
random_final_results = aggregate_results_across_seeds(random_results_across_seeds)

ppo_profit = calculate_profit(ppo_final_results['mean_yield'], ppo_final_results['mean_irrigation'])
random_profit = calculate_profit(random_final_results['mean_yield'], random_final_results['mean_irrigation'])
ppo_water_efficiency = calculate_water_efficiency(ppo_final_results['mean_yield'], ppo_final_results['mean_irrigation'])
random_water_efficiency = calculate_water_efficiency(random_final_results['mean_yield'], random_final_results['mean_irrigation'])

ppo_df = pd.DataFrame({
    'Dry yield (tonne/ha)_mean': [ppo_final_results['mean_yield']],
    'Dry yield (tonne/ha)_std': [ppo_final_results['std_yield']],
    'Seasonal irrigation (mm)_mean': [ppo_final_results['mean_irrigation']],
    'Seasonal irrigation (mm)_std': [ppo_final_results['std_irrigation']],
    'Profit_mean': [ppo_profit],
    'WaterEfficiency_mean': [ppo_water_efficiency]
}, index=['PPO'])

random_df = pd.DataFrame({
    'Dry yield (tonne/ha)_mean': [random_final_results['mean_yield']],
    'Dry yield (tonne/ha)_std': [random_final_results['std_yield']],
    'Seasonal irrigation (mm)_mean': [random_final_results['mean_irrigation']],
    'Seasonal irrigation (mm)_std': [random_final_results['std_irrigation']],
    'Profit_mean': [random_profit],
    'WaterEfficiency_mean': [random_water_efficiency]
}, index=['Random'])

# Concatenate all results
comparison_df = pd.concat([ppo_df, random_df, aquacrop_results_agg])
comparison_df.to_csv(os.path.join(eval_output_dir, 'comparison_results.csv'))

# Plotting
combined_labels = ['PPO', 'Thresholds', 'Interval', 'Net', 'Rainfed', 'Random']
combined_yields = [
    ppo_final_results['mean_yield'],
    aquacrop_results_agg.loc['Thresholds', 'Dry yield (tonne/ha)_mean'],
    aquacrop_results_agg.loc['Interval', 'Dry yield (tonne/ha)_mean'],
    aquacrop_results_agg.loc['Net', 'Dry yield (tonne/ha)_mean'],
    aquacrop_results_agg.loc['Rainfed', 'Dry yield (tonne/ha)_mean'],
    random_final_results['mean_yield']
]
combined_irrigations = [
    ppo_final_results['mean_irrigation'],
    aquacrop_results_agg.loc['Thresholds', 'Seasonal irrigation (mm)_mean'],
    aquacrop_results_agg.loc['Interval', 'Seasonal irrigation (mm)_mean'],
    aquacrop_results_agg.loc['Net', 'Seasonal irrigation (mm)_mean'],
    aquacrop_results_agg.loc['Rainfed', 'Seasonal irrigation (mm)_mean'],
    random_final_results['mean_irrigation']
]
combined_profits = [
    ppo_profit,
    aquacrop_results_agg.loc['Thresholds', 'Profit_mean'],
    aquacrop_results_agg.loc['Interval', 'Profit_mean'],
    aquacrop_results_agg.loc['Net', 'Profit_mean'],
    aquacrop_results_agg.loc['Rainfed', 'Profit_mean'],
    random_profit
]
combined_water_efficiency = [
    ppo_water_efficiency,
    aquacrop_results_agg.loc['Thresholds', 'WaterEfficiency_mean'],
    aquacrop_results_agg.loc['Interval', 'WaterEfficiency_mean'],
    aquacrop_results_agg.loc['Net', 'WaterEfficiency_mean'],
    aquacrop_results_agg.loc['Rainfed', 'WaterEfficiency_mean'],
    random_water_efficiency
]
combined_yields_std = [
    ppo_final_results['std_yield'],
    aquacrop_results_agg.loc['Thresholds', 'Dry yield (tonne/ha)_std'],
    aquacrop_results_agg.loc['Interval', 'Dry yield (tonne/ha)_std'],
    aquacrop_results_agg.loc['Net', 'Dry yield (tonne/ha)_std'],
    aquacrop_results_agg.loc['Rainfed', 'Dry yield (tonne/ha)_std'],
    random_final_results['std_yield']
]
combined_irrigations_std = [
    ppo_final_results['std_irrigation'],
    aquacrop_results_agg.loc['Thresholds', 'Seasonal irrigation (mm)_std'],
    aquacrop_results_agg.loc['Interval', 'Seasonal irrigation (mm)_std'],
    aquacrop_results_agg.loc['Net', 'Seasonal irrigation (mm)_std'],
    aquacrop_results_agg.loc['Rainfed', 'Seasonal irrigation (mm)_std'],
    random_final_results['std_irrigation']
]

colors = ['gold', 'blue', 'green', 'red', 'purple', 'gray']

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_yields, yerr=combined_yields_std, capsize=5, color=colors, edgecolor='black')
plt.ylabel('Mean yield (tonne/ha)')
plt.title('Comparison of mean yields')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_yields.png'), dpi=300)

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_irrigations, yerr=combined_irrigations_std, capsize=5, color=colors, edgecolor='black')
plt.ylabel('Total irrigation (mm)')
plt.title('Comparison of total irrigation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_irrigations.png'), dpi=300)

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_profits, color=colors, edgecolor='black')
plt.ylabel('Profit ($)')
plt.title('Comparison of profits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_profits.png'), dpi=300)

plt.figure(figsize=(12, 7))
plt.bar(combined_labels, combined_water_efficiency, color=colors, edgecolor='black')
plt.ylabel('Water efficiency (kg/m³)')
plt.title('Comparison of water efficiency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_output_dir, 'combined_water_efficiency.png'), dpi=300)

print("Evaluation completed.")
