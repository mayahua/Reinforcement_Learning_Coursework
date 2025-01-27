import copy
import pickle
from collections import defaultdict

import gymnasium as gym
from gymnasium import Space
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2024.constants import EX5_RACETRACK_CONSTANTS as RACETRACK_CONSTANTS

from rl2024.exercise4.agents import DDPG
from rl2024.exercise4.train_ddpg import train
from rl2024.exercise3.replay import ReplayBuffer
from rl2024.util.hparam_sweeping import generate_hparam_configs, grid_search, random_search
from rl2024.util.result_processing import Run


RENDER = False
SWEEP = True # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 3 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = True # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "RACETRACK"

# IN EXERCISE 5 YOU SHOULD TUNE PARAMETERS IN THIS CONFIG ONLY
RACETRACK_CONFIG = {
    "policy_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [63, 59, 63], 
    "policy_hidden_size": [65, 63, 59], 
    "gamma": 0.93,
    "tau": 0.005,
    "batch_size": 70,
    "buffer_capacity": int(1e6),
}
RACETRACK_CONFIG.update(RACETRACK_CONSTANTS)

### INCLUDE YOUR CHOICE OF HYPERPARAMETERS HERE ###
'''
critic_hidden_size_values = []
for _ in range(3):
    hidden_sizes = [round(random_search(1, distribution='uniform', min=50, max=70).item()) for _ in range(3)]
    critic_hidden_size_values.append(hidden_sizes)

policy_hidden_size_values = []
for _ in range(3):
    hidden_sizes = [round(random_search(1, distribution='uniform', min=50, max=70).item()) for _ in range(3)]
    policy_hidden_size_values.append(hidden_sizes)

critic_learning_rate_values = random_search(3, 'exponential', 1e-4, 1e-2)
policy_learning_rate_values = random_search(3, 'exponential', 1e-4, 1e-2)

gamma_values = random_search(3, 'exponential', 9e-1, 9.9e-1)

tau_values = random_search(3, 'exponential', 5e-4, 5e-2)

batch_size_values = [int(value) for value in grid_search(50, 80, 3)]
'''

critic_hidden_size_values = [[63, 59, 63]]
policy_hidden_size_values = [[65, 63, 59]]
critic_learning_rate_values = [1e-3]
policy_learning_rate_values = [1e-3]
gamma_values = [9.3e-1]
tau_values = [5e-3]
batch_size_values = [70]

RACETRACK_HPARAMS = {
    "critic_hidden_size": critic_hidden_size_values,
    "policy_hidden_size": policy_hidden_size_values,
    "critic_learning_rate": critic_learning_rate_values,
    "policy_learning_rate": policy_learning_rate_values,
    "gamma": gamma_values,
    "tau": tau_values,
    "batch_size": batch_size_values,
}

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Racetrack-sweep-results-ex5.pkl"

if __name__ == "__main__":
    if ENV == "RACETRACK":
        CONFIG = RACETRACK_CONFIG
        HPARAMS_SWEEP = RACETRACK_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_BIPEDAL
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    env_eval = gym.make(CONFIG["env"])

    if SWEEP and HPARAMS_SWEEP is not None:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run...")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i + 1}/{NUM_SEEDS_SWEEP}")
                print(f"{hparams_values}")                
                run_save_filename = '--'.join([run.config["algo"], run.config["env"], str(i)])#, hparams_values
                # Filename changed (hyperparameter values excluded) to avoid failing to save files (see Piazza question @84 & @121). I tried substitute the ':' but still fail, so I suppose it may because the filenames are too long since I have multiple hyperparameters.
                if SWEEP_SAVE_ALL_WEIGTHS:
                    run.set_save_filename(run_save_filename)
                eval_returns, eval_timesteps, times, run_data = train(env, env_eval, run.config, output=True)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")

        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)

    else:
        raise NotImplementedError('You are attempting to run normal training within the hyperparameter tuning file!')

    env.close()
