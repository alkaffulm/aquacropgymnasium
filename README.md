# AquaCropGymnasium

## Introduction

This repository accompanies the paper **"Optimizing Water Use in Maize Irrigation with Reinforcement Learning"** by Muhammad Alkaff, Yuslena Sari, and Abdullah Basuhail. It presents **AquaCropGymnasium**, a reinforcement learning (RL) environment designed to optimize irrigation strategies by integrating [AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/) with the [Gymnasium](https://gymnasium.farama.org/) framework. This project builds upon the [AquaCrop-Gym](https://github.com/aquacropos/aquacrop-gym) repository, updated to be compatible with AquaCrop-OSPy **3.0.9** and Gymnasium **0.29**.

[AquaCrop](https://www.fao.org/aquacrop/en/) is a crop growth model developed by the FAO's Land and Water Division. It balances accuracy, robustness, and simplicity, making it well-suited for simulating the yield response of herbaceous crops to varying water availability, especially under water-limited conditions.

[AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/) provides a Python-based interface to AquaCrop, enabling advanced research in crop-water management. It empowers users to design and test irrigation strategies with minimal programming complexity.

By integrating AquaCrop-OSPy into a comprehensive RL environment, AquaCropGymnasium facilitates the development and evaluation of intelligent irrigation policies. This approach contributes to sustainable agriculture by leveraging RL to optimize water use while maintaining crop productivity and profitability.

## Features

- **Updated Compatibility:** Works with AquaCrop-OSPy **3.0.9** and Gymnasium **0.29**.
- **Reinforcement Learning Integration:** Easily train and evaluate RL algorithms (e.g., PPO) using Stable-Baselines3.
- **Streamlined Setup:** Manages dependencies and installation via Poetry for a simpler setup process.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/alkaffulm/aquacropgymnasium.git
    cd aquacropgymnasium
    ```

2. **Install Poetry:**
    If Poetry is not installed, please follow the [Poetry Installation Instructions](https://python-poetry.org/docs/#installation).

3. **Install Dependencies:**
    ```bash
    poetry install
    ```
    This command creates a virtual environment and installs all required dependencies.

4. **Activate the Virtual Environment:**
    ```bash
    poetry shell
    ```

AquaCropGymnasium is now ready for training and evaluation of RL-based irrigation strategies.

## Training and Evaluation

- **Train a PPO Model:**
    ```bash
    python train.py
    ```

- **Evaluate a Trained PPO Model:**
    ```bash
    python evaluate.py
    ```

- **Combined Training and Evaluation:**
    ```bash
    python train.py && python evaluate.py
    ```

## Troubleshooting

If you encounter issues (e.g., missing modules, compilation errors) during installation or execution, try running AquaCrop in development mode. This approach disables certain function compilations, potentially increasing runtime but improving compatibility:

```python
import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'
```

For further guidance, please visit the [AquaCrop-OSPy GitHub repository](https://github.com/aquacropos/aquacrop?tab=readme-ov-file#installation-troubleshooting).
