import optuna
import matplotlib.pyplot as plt
from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil
from aquacrop.utils import get_filepath, prepare_weather

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

sim_start = "2008/05/01"
sim_end = "2018/12/31"
soil = Soil('SandyLoam')
crop_obj = Crop('Maize', planting_date='05/01')
initWC = InitialWaterContent(value=['FC'])

def run_model(smts, max_irr_season, year1, year2):
    irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrrSeason=max_irr_season)
    model = AquaCropModel(f'{year1}/05/01', f'{year2}/10/31', wdf, soil, crop_obj, 
                          irrigation_management=irrmngt, initial_water_content=initWC)
    model.run_model(till_termination=True)
    return model.get_simulation_results()

def evaluate(smts, max_irr_season, test=False):
    out = run_model(smts, max_irr_season, year1=2008, year2=2018)
    yld = out['Dry yield (tonne/ha)'].mean()
    tirr = out['Seasonal irrigation (mm)'].mean()
    reward = yld
    return (yld, tirr, reward) if test else -reward

def objective(trial):
    smt1 = trial.suggest_uniform('smt1', 0.0, 100.0)
    smt2 = trial.suggest_uniform('smt2', 0.0, 100.0)
    smt3 = trial.suggest_uniform('smt3', 0.0, 100.0)
    smt4 = trial.suggest_uniform('smt4', 0.0, 100.0)

    smts = [smt1, smt2, smt3, smt4]
    reward = evaluate(smts, max_irr_season=300)
    
    return reward

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_trial = study.best_trial
print("Best SMT values:")
print(f"SMT 1: {best_trial.params['smt1']}")
print(f"SMT 2: {best_trial.params['smt2']}")
print(f"SMT 3: {best_trial.params['smt3']}")
print(f"SMT 4: {best_trial.params['smt4']}")