# Webots RL Navigation with PPO & REINFORCE

A reinforcement learning framework for robotic navigation in Webots using PPO (Stable-Baselines3) and a custom REINFORCE algorithm. The robot uses camera-based lane detection and LiDAR to navigate to a goal while avoiding obstacles. Designed for simulation and transferable to AWS DeepRacer.

---

## Key Features

- **Navigation Task**  
  Navigate to a goal while avoiding obstacles—using LiDAR for obstacles and camera for lane following.
- **RL Algorithms**  
  - PPO (Stable-Baselines3)  
  - Custom REINFORCE with baseline support
- **Sensor Suite**  
  GPS, LiDAR, compass, gyro, accelerometer, touch & distance sensors, plus camera.
- **Image Processing**  
  Canny → Hough → perspective warp → steering-angle computation.
- **Experiment Tracking**  
  Weights & Biases for hyperparameter sweeps; TensorBoard for logs.
- **Configurable**  
  Manual hyperparameters or fully automated W&B sweeps.
- **Real-World Adaptability**  
  Seamless transfer to AWS DeepRacer’s camera, IMU, and motor interface.

---

## Tech Stack

| Component            | Library / Tool                  |
|----------------------|---------------------------------|                          
| RL                   | Stable-Baselines3 (PPO), Custom REINFORCE |
| Simulation           | Webots; AWS DeepRacer (real)    |
| Image Processing     | OpenCV, NumPy, Matplotlib       |
| Logging              | Weights & Biases, TensorBoard   |
| DL Framework         | PyTorch                         |
| Env Wrappers         | Gymnasium                       |
| Utilities            | Git, datetime, os               |

---

## Workflow

![Workflow Diagram](attachment://WorkFlow.png)

1. **Sensors** (LiDAR & Camera) feed raw data →  
2. **Pre-processing Nodes**  
   - LiDAR → obstacle detection  
   - Camera → lane detection (Canny, Hough, warp)  
3. **Planning Node** (fuses LiDAR + lanes) → path plan  
4. **Control Node** → motor/steering commands  
5. **Simulator & Robot** (Webots or DeepRacer hardware)  
6. **Feedback Loop**: sensor → plan → control → act → sensor…

---

## Setup

1. **Clone repo**  
   ```bash
   git clone https://github.com/elhamarmin/Autonomous-Deepracer.git
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   pip install webots stable-baselines3 gymnasium wandb torch opencv-python matplotlib numpy
   ```
3. **Install Webots**  
   - Download from https://cyberbotics.com/  
   - Add its Python API to `PYTHONPATH`.
4. **Configure**  
   - Place your Webots world and robot model (4-wheel + sensors).  
   - Adjust `destination_coordinate` in `WebotsEnvironment.py`.
5. **Run**  
   No build needed—just execute your training/testing scripts.

---

## Usage

### Manual Training (PPO)
```python
from your_script import manual_train_model

manual_train_model(
    learning_rate=3e-4,
    total_timesteps=30_000,
    ent_coef=0.15,
    clip_range=0.1478,
    gae_lambda=0.7755,
    n_epochs=64,
    n_steps=1024,
    batch_size=64,
)
```
Model saved to `./trained_model/model_<timestamp>.zip`.

### Automated Sweeps (Weights & Biases)
```python
from your_script import train_model, sweep_configuration
import wandb

sweep_id = wandb.sweep(sweep_configuration, project="sb3")
wandb.agent(sweep_id, function=train_model, count=30)
```

### REINFORCE Training
```python
from your_script import Agent_REINFORCE, Policy_Network, WebotsGymEnv
import torch

env    = WebotsGymEnv()
policy = Policy_Network(3, 6, 3).to(device)
agent  = Agent_REINFORCE(
    policy_network=policy,
    environment=env,
    save_path='./results',
    load_path='./results/best_weights.pt',
    num_episodes=2000,
    max_steps=500,
    learning_rate=1e-4,
    gamma=0.9,
    clip_grad_norm=5,
    baseline=True,
)
agent.train()
```
Plots saved under `./results/reward_plot.png`.

### Testing & Debugging
- **Test**:  
  ```python
  agent.test()
  ```
- **Debug** (manual control): set `run_model = 2` and use `w/s/a/d/i` to move.

- **TensorBoard**:  
  ```bash
  tensorboard --logdir logs/train
  ```

---

## Lane Detection System

Pipeline steps:

1. **ROI Crop**  
   Trapezoid mask via `region_of_interest`.
2. **Color Filter**  
   Yellow range in BGR → grayscale.
3. **Perspective Warp**  
   Bird’s-eye via `perspective_transform`.
4. **Edge & Line Detection**  
   Canny & HoughLinesP.
5. **Weight Computation**  
   Classify left/right by x-coords → compute `left_weight`, `right_weight`.
6. **Steering Angle**  
   `(right_weight – left_weight) / (image_width/2)`.

Outputs all intermediate images and weights for debugging.

---

## Real-World Deployment (AWS DeepRacer)

1. **Hardware**  
   - Match DeepRacer’s camera (1080×720).  
   - Map LiDAR to IMU or external LiDAR.
2. **Software**  
   ```bash
   pip install awscam
   ```
   - Use AWS RoboMaker or local sim.  
   - Adapt `WebotsEnvironment` to ROS topics (`/sensors/camera/image_raw`, `/cmd_vel`).
3. **Workflow Mapping**  
   LiDAR → IMU, Camera → pipeline, Planning → reuse, Control → ROS.
4. **Train & Upload**  
   - Export Webots model.  
   - Upload ZIP to AWS DeepRacer console.  
   - Monitor via CloudWatch.

**Considerations**  
- Calibrate for real-world lighting.  
- Minimize network latency.  
- Implement emergency stop.

---

## Configuration

- **PPO**: `learning_rate`, `ent_coef`, `clip_range`, `gae_lambda`, `n_steps`, `batch_size`, `n_epochs`, `total_timesteps`.
- **REINFORCE**: `num_episodes`, `max_steps`, `learning_rate`, `gamma`, `clip_grad_norm`, `baseline`.
- **Env**:  
  - `destination_coordinate` (default: `[-3.13, 0]`)  
  - `reach_threshold`: 0.1  
  - `safe_distance`: 0.35
- **Lane**: `center_margin`, color thresholds, Canny params.

---

## Results

- **Models**:  
  `./trained_model/model_<id>_<ts>.zip`  
  `./results/ep_vis_<episode>.png`
- **Logs**:  
  W&B: `hit_count`, `goal_count`, loss, gradients  
  TensorBoard: `logs/train`, `logs/fit`
- **Visuals**:  
  - Reward curves (`reward_plot.png`)  
  - Canny & warped images  
  - Interactive network graphs

