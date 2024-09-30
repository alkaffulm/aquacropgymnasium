import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from aquacrop.core import AquaCropModel
from aquacrop.entities.crop import Crop
from aquacrop.entities.inititalWaterContent import InitialWaterContent
from aquacrop.entities.irrigationManagement import IrrigationManagement
from aquacrop.entities.soil import Soil
from aquacrop.utils import prepare_weather

class Maize(gym.Env):
    def __init__(
        self,
        render_mode=None,
        mode='train',
        year1=1982,
        year2=2018,
        crop='Maize',
        climate_file=None,
        planting_date=None
    ):
        super(Maize, self).__init__()
        self.year1 = year1
        self.year2 = year2
        self.init_wc = InitialWaterContent(value=['FC'])
        self.crop_name = crop
        self.climate = climate_file if climate_file is not None else 'champion_climate.txt'

        base_path = os.path.dirname(os.path.dirname(__file__))
        self.climate_file_path = os.path.abspath(os.path.join(base_path, 'weather_data', self.climate))

        self.planting_date = planting_date if planting_date is not None else '05/01'
        self.soil = Soil('SandyLoam')

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        self.action_depths = [0, 25]
        self.action_space = spaces.Discrete(len(self.action_depths))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year

        self.crop = Crop(self.crop_name, self.planting_date)

        try:
            self.wdf = prepare_weather(self.climate_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Climate file not found at {self.climate_file_path}")

        self.wdf['Year'] = self.simcalyear
        self.total_irrigation_applied = 0
        self.cumulative_reward = 0

        self.model = AquaCropModel(
            f'{self.simcalyear}/{self.planting_date}',
            f'{self.simcalyear}/12/31',
            self.wdf,
            self.soil,
            self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=5),
            initial_water_content=self.init_wc
        )

        self.model.run_model()

        obs = self._get_obs()
        info = {'dry_yield': 0.0, 'total_irrigation': 0}
        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond
        precip_last_7_days = self._get_last_7_days_values('Precipitation')
        min_temp_last_7_days = self._get_last_7_days_values('MinTemp')
        max_temp_last_7_days = self._get_last_7_days_values('MaxTemp')
        weather_obs = np.concatenate([precip_last_7_days, min_temp_last_7_days, max_temp_last_7_days])

        obs = np.array([
            cond.age_days,
            cond.canopy_cover,
            cond.biomass,
            cond.depletion,
            cond.taw
        ], dtype=np.float32)

        obs = np.concatenate([obs, weather_obs])
        return obs

    def _get_last_7_days_values(self, column):
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day][column]
        if len(last_7_days) < 7:
            padding = np.zeros(7 - len(last_7_days))
            last_7_days = np.concatenate([padding, last_7_days])
        return last_7_days

    def step(self, action):
        depth = self.action_depths[int(action)]
        self.model._param_struct.IrrMngt.depth = depth
        self.model.run_model(initialize_model=False)
        next_obs = self._get_obs()
        terminated = self.model._clock_struct.model_is_finished
        self.total_irrigation_applied += depth
        step_reward = 0
        if depth > 0:
            step_reward -= self.total_irrigation_applied
        self.cumulative_reward += step_reward

        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            total_irrigation = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()
            yield_reward = dry_yield ** 4
            self.cumulative_reward += yield_reward
            info = {'dry_yield': dry_yield, 'total_irrigation': total_irrigation}
            total_reward = self.cumulative_reward
            self.cumulative_reward = 0
        else:
            info = {'dry_yield': 0.0, 'total_irrigation': self.total_irrigation_applied}
            total_reward = step_reward
        return next_obs, total_reward, terminated, False, info

    def close(self):
        pass
