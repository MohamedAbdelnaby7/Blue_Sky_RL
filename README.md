# Deep Reinforcement Learning for Air Traffic Control

This repository contains an implementation of 3D air traffic control reinforcement learning environments using the BlueSky simulator, extending the research from "BlueSkyGym: Reinforcement Learning Environments for Air Traffic Applications."

## Project Overview

This project extends existing Gymnasium-based air traffic control (ATC) environments to include vertical speed as an additional action dimension. It then uses Deep Reinforcement Learning (DRL) algorithms (PPO and DDPG) to train agents in these environments.

Key features:
- Extension of SectorCREnv and StaticObstacleEnv to include vertical speed control
- Observation space augmented with altitude difference information
- Training implementation using Stable-Baselines3
- Evaluation and video recording capabilities

## Prerequisites

- Python 3.9 or higher
- Conda (recommended for environment management)
- Git

## Installation

### 1. Create a Conda Environment

```bash
conda create -n atc python=3.10
conda activate atc
```

### 2. Clone this Repo

```bash
git clone https://github.com/junzis/bluesky-gym.git](https://github.com/MohamedAbdelnaby7/Blue_Sky_RL.git
cd bluesky-gym
git checkout main_bluesky
pip install -e .
```

### 3. CD back to our code

```bash
cd ..
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

## Project Structure

```
rbe595-atc-rl/
├── envs_patch/
│   ├── __init__.py
│   ├── register_v1_envs.py
│   ├── sector_cr_env_v1.py
│   ├── static_obstacle_env_v1.py
├── models/                   # Directory for saved models (will be created)
├── videos/                   # Directory for output videos (will be created)
├── train_ppo.py              # PPO training script
├── train_ddpg.py             # DDPG training script
├── record_video.py           # Script for creating videos of trained agents
├── compatible_evaluate_agent.py   # Script for evaluating trained agents
├── requirements.txt
└── README.md
```

## Environment Extensions

This project extends the original environments with the following enhancements:

1. SectorCREnvV1:
   - 3D continuous action vector: [Δ-heading (deg), Δ-speed (kts), vertical_speed (ft/min)]
   - Observation augmented with altitude-delta (own_alt − intruder_alt) per intruder

2. StaticObstacleEnvV1:
   - Also includes vertical_speed as an action dimension
   - Observation includes altitude differences with obstacles
   - Handles collisions in 3D space

## Usage

### Training an Agent

#### Using PPO:

```bash
python train_ppo.py --env SectorCREnv-v1 --steps 3000000
```

or

```bash
python train_ppo.py --env StaticObstacleEnv-v1 --steps 3000000
```

#### Using DDPG:

```bash
python train_ddpg.py --env SectorCREnv-v1 --steps 3000000
```

or

```bash
python train_ddpg.py --env StaticObstacleEnv-v1 --steps 3000000
```

### Evaluating a Trained Agent

```bash
python compatible_evaluate_agent.py --model models/SectorCREnv-v1-ppo --algo ppo --env SectorCREnv-v1 --episodes 5
```

### Recording a Video of Agent Performance

```bash
python record_video.py --model models/StaticObstacleEnv-v1-ppo --env StaticObstacleEnv-v1 --out videos/obstacle_demo.mp4
```

or using the evaluation script with recording:

```bash
python compatible_evaluate_agent.py --model models/SectorCREnv-v1-ppo --algo ppo --env SectorCREnv-v1 --record --out videos/sector_demo.mp4
```

## Troubleshooting

### Common Issues

1. Environment Creation Errors:
   - Make sure you've imported the environments correctly with `import envs_patch.register_v1_envs`
   - Verify that BlueSky-Gym is properly installed

2. Rendering Issues:
   - The BlueSky simulator can sometimes have rendering issues. Try using the `record_video.py` script which is specifically designed to work with these environments.

3. Memory Errors During Training:
   - Reduce batch size or use a machine with more RAM

4. Cleanup Errors:
   - Errors like `AttributeError: 'NoneType' object has no attribute 'areatree'` during cleanup are known issues with the BlueSky simulator and generally don't affect training results

## Notes on Training

- Training can take several hours depending on your hardware
- Models are automatically saved in the `models/` directory
- Training progress can be monitored using TensorBoard:
  ```bash
  tensorboard --logdir tb
  ```
