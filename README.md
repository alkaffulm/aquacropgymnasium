# AquaCropGymnasium

## Introduction

In our paper, **"Sustainable agriculture through efficient irrigation strategies using reinforcement learning"** by Muhammad Alkaff, Yuslena Sari, and Abdullah Basuhail, we introduce AquaCropGymnasium, a reinforcement learning (RL) environment developed to optimize irrigation strategies by integrating [AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/) with the [Gymnasium](https://gymnasium.farama.org/) framework. This project adapts the [AquaCrop-Gym](https://github.com/aquacropos/aquacrop-gym) repository, updated to support AquaCrop-OSPy version **3.0.9** and Gymnasium version **0.29**.

[AquaCrop](https://www.fao.org/aquacrop/en/), developed by the Land and Water Division of FAO, is a crop model that simulates the yield response of herbaceous crops to water availability. It is designed to balance simplicity, accuracy, and robustness, making it a widely utilized tool for assessing the impacts of water management on crop production, particularly under water-limited conditions.

[AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/) is the Python implementation of AquaCrop, derived from the open-source AquaCrop-OS model. It facilitates advanced research in crop-water management by enabling users to design and test irrigation strategies with minimal programming expertise.

AquaCropGymnasium integrates these models into a comprehensive RL environment, allowing for the development and evaluation of intelligent irrigation strategies, thereby contributing to research in sustainable agriculture.


## Features

- **Compatibility**: Supports AquaCrop-OSPy **3.0.9** and Gymnasium **0.29**.
- **RL Algorithm Integration**: Integrates with Stable-Baselines3 for training and evaluation.
- **Simplified Setup**: Uses Poetry for easy installation and management.

## Usage

### Installation

To set up AquaCropGymnasium, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/aquacropgymnasium.git
    cd aquacropgymnasium
    ```

2. **Install Poetry**:

    If Poetry is not already installed, follow the instructions in the [Poetry README](https://github.com/python-poetry/poetry#installation) to install it.

3. **Install dependencies using Poetry**:

    Once Poetry is installed, run:

    ```bash
    poetry install
    ```

    This command will create a virtual environment and install all necessary dependencies.

4. **Activate the virtual environment**:

    After installation, activate the virtual environment with:

    ```bash
    poetry shell
    ```

AquaCropGymnasium is now ready for use. You can start training and evaluating reinforcement learning models for irrigation strategies.

### Training and Evaluation

- To **train** a PPO model, run:

    ```bash
    python train.py
    ```

- To **evaluate** a trained PPO model, run:

    ```bash
    python evaluate.py
    ```

- To run **both training and evaluation** sequentially, use:

    ```bash
    python train.py && python evaluate.py
    ```

### Troubleshooting

If you encounter issues during installation, such as missing modules or compilation errors, you can run AquaCrop in development mode. This approach disables certain function compilations, resulting in slightly increased runtime but ensuring compatibility:

```python
import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'
```

For further guidance, please visit the [AquaCrop-OSPy GitHub repository](https://github.com/aquacropos/aquacrop?tab=readme-ov-file#installation-troubleshooting).