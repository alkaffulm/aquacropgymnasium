## Introduction

This repository provides the implementation of **AquaCropGymnasium**, a reinforcement learning (RL) environment designed for optimizing irrigation strategies by integrating [AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/) with the [Gymnasium](https://gymnasium.farama.org/) framework. It builds upon the [AquaCrop-Gym](https://github.com/aquacropos/aquacrop-gym) repository and is updated for compatibility with AquaCrop-OSPy **3.0.9** and Gymnasium **0.29**.

[AquaCrop](https://www.fao.org/aquacrop/en/) is a crop growth model developed by the FAO's Land and Water Division to simulate the yield response of herbaceous crops under varying water availability. [AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/) provides a Python-based interface for AquaCrop, facilitating research in crop-water management and irrigation scheduling.

AquaCropGymnasium integrates AquaCrop-OSPy with a reinforcement learning framework, allowing the development and evaluation of RL-based irrigation policies. This environment supports RL algorithms such as Proximal Policy Optimization (PPO) and enables systematic comparisons between RL-based and traditional irrigation strategies.

## Features

- **Updated Compatibility:** Supports AquaCrop-OSPy **3.0.9** and Gymnasium **0.29**.
- **Reinforcement Learning Integration:** Provides an RL training environment compatible with Stable-Baselines3.
- **Streamlined Setup:** Uses Poetry for dependency management and installation.

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

## Citing AquaCropGymnasium

If you use **AquaCropGymnasium** in your research, please cite our paper:

**Muhammad Alkaff, Abdullah Basuhail, Yuslena Sari**  
*"Optimizing Water Use in Maize Irrigation with Reinforcement Learning"*  
**Mathematics**, Volume 13, Issue 4, Article 595, 2025  
[DOI: 10.3390/math13040595](https://www.mdpi.com/2227-7390/13/4/595)
