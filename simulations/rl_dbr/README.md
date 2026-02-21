# Reinforcement Learning-based Drum-Buffer-Rope (RL-DBR) Scheduler

This folder contains a reinforcement learning-based scheduler for a manufacturing environment, built on the Drum-Buffer-Rope (DBR) methodology. The system uses a Proximal Policy Optimization (PPO) agent from the `stable-baselines3` library to make production scheduling decisions.

## Core Concepts

### Drum-Buffer-Rope (DBR)

DBR is a production scheduling methodology that aims to maximize throughput by focusing on the system's constraint.

*   **Drum**: The system's constraint, or bottleneck, which dictates the pace of the entire production line. In this implementation, the constraint resource is automatically identified based on the product processing times and demand.
*   **Buffer**: A time-based buffer of work-in-process (WIP) is maintained in front of the constraint to ensure it is never starved for work. This protects the system from variability. The environment tracks `constraint_buffer_wip` and `constraint_buffer_queue`.
*   **Rope**: The mechanism that releases new work into the system. The "rope" is tied to the drum, meaning that new work is only released at the rate that the drum can process it. In this project, the reinforcement learning agent acts as the rope, deciding how much of each product to release.

### Reinforcement Learning (RL) Agent

A PPO agent is trained to make intelligent scheduling decisions.

*   **Observation (State)**: The agent observes the state of the factory, which includes:
    *   The level of the constraint buffer (WIP and queue).
    *   The amount of WIP for each product.
    *   The amount of finished goods for each product.
*   **Action**: The agent's action is to decide the quantity of each product to add to the production schedule for the next interval.
*   **Reward**: The reward function is designed to incentivize the agent to:
    *   Minimize work-in-process (WIP) and finished goods inventory to reduce holding costs.
    *   Minimize lost sales to maximize customer satisfaction and revenue.

## Files

*   `environment.py`: Defines the `DBRLEnv`, a `gymnasium` environment that simulates the manufacturing plant and the DBR methodology. It's built on top of the `manusim` simulation engine.
*   `model_training.py`: A script for training the PPO agent. It uses `hydra` for configuration and saves the trained model.
*   `simulation.py`: A script for running a simulation. It can either use a pre-trained model for inference or run in a "training" mode where an external process (like `model_training.py`) controls the actions.
*   `config/`: This directory contains the configuration files (using `hydra`) for training and simulation. This is where you can define the factory layout, products, demand, and training parameters.
*   `models/`: This directory is the default location where trained models are saved.

## How to Run

The project uses `hydra` for configuration management.

### Training a Model

To train a new model, run the `model_training.py` script. You can override the default configuration parameters from the command line.

```bash
python model_training.py
```

Or with custom parameters:

```bash
python model_training.py training.n_envs=8 training.total_timesteps=1000000
```

### Running a Simulation

To run a simulation with a trained model, use the `simulation.py` script. You need to specify the path to the trained model in the `simulation_config.yaml` or override it from the command line.

```bash
python simulation.py
```

Make sure the `simulation.model_path` in `config/simulation_config.yaml` points to a valid model directory.
