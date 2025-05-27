
# DEEPRACER AUTOMOUNOS: RL-Based Lane-Aware Navigation in Webots

DEEPRACER AUTOMOUNOS is an end-to-end reinforcement learning framework designed for autonomous robotic navigation in Webots. It combines sensor fusion (camera, LiDAR, GPS, IMU) with advanced lane detection and state-aware control using Proximal Policy Optimization (PPO) and a custom REINFORCE algorithm. The agent learns to reach a goal while avoiding obstacles, guided by real-time camera vision and geometric lane estimation.

---

## Features

- **Autonomous Navigation**: Trains agents to reach destination goals while avoiding obstacles using state estimation and lane-based control.
- **Dual RL Algorithms**: Supports PPO (via Stable-Baselines3) and a custom REINFORCE algorithm with optional baseline.
- **Rich Sensor Suite**: Integrates GPS, LiDAR, camera, IMU, distance sensors, and more.
- **Camera-Based Lane Detection**: Uses OpenCV to compute a steering angle from real-time images.
- **Experiment Management**: Logs training runs via Weights & Biases and TensorBoard.
- **Flexible Training**: Run manual sessions with fixed hyperparameters or sweep-based experiments.

---

## 🔧 Tech Stack

| Module              | Tool / Library                   |
|---------------------|----------------------------------|
| Simulation          | Webots                           |
| RL Algorithms       | PPO (SB3), Custom REINFORCE      |
| Vision & Detection  | OpenCV, NumPy, Matplotlib        |
| Logging             | Weights & Biases, TensorBoard    |
| Deep Learning       | PyTorch                          |
| Environment Wrapping| Gymnasium                        |
| Language            | Python                           |

---

## Setup

1. **Clone the Repository**
   ```bash
   git clone [Autonomous-Deepracer](https://github.com/elhamarmin/Autonomous-Deepracer.git)
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Webots**
   - [Download Webots](https://cyberbotics.com/)
   - Ensure the Python controller is configured (`PYTHONPATH` includes Webots `lib` directory)

4. **Configure Environment**
   - Use the provided world file with the robot model and sensors.
   - Optionally adjust `destination_coordinate` in `WebotsEnvironment.py`.

---

## Training

### 1. Manual PPO Training
```python
from your_script import manual_train_model

manual_train_model(
    learning_rate=3e-4,
    total_timesteps=30000,
    ent_coef=0.15,
    clip_range=0.1478,
    gae_lambda=0.7755,
    n_epochs=64,
    n_steps=1024,
    batch_size=64
)
```

### 2. Weights & Biases Sweep
```python
from your_script import train_model, sweep_configuration

sweep_id = wandb.sweep(sweep=sweep_configuration, project="sb3")
wandb.agent(sweep_id, function=train_model, count=30)
```

### 3. Custom REINFORCE
```python
from your_script import Agent_REINFORCE, Policy_Network, WebotsGymEnv

env = WebotsGymEnv()
policy = Policy_Network(input_size=3, hidden_size=6, output_size=3).to(device)

agent = Agent_REINFORCE(
    policy_network=policy,
    environment=env,
    num_episodes=2000,
    learning_rate=1e-4,
    gamma=0.9,
    baseline=True,
    clip_grad_norm=5
)
agent.train()
```

---

## Evaluation & Debugging

- **Evaluate**:
  ```python
  agent.test()
  ```

- **Debug**: Set `run_model = 2` to activate manual control (WASD keys + `i` for stepping). Prints LiDAR data for obstacle proximity.

---

## Lane Detection Pipeline

The vision system extracts lane features and computes a steering angle to guide the robot.

### Key Stages:

- **Region of Interest (ROI)**:
  - Extracts trapezoidal region focused on the road.
  - Suppresses irrelevant parts using a polygonal mask.

- **Color Filtering**:
  - Filters yellow lanes with a tuned BGR range.
  - Converts the result to grayscale for further processing.

- **Perspective Transform**:
  - Warps the ROI to a bird's-eye view.
  - Enables geometric simplification of lane geometry.

- **Edge & Line Detection**:
  - Canny edge detection with thresholds (850, 250).
  - HoughLinesP for line segment detection.

- **Lane Separation & Weighting**:
  - Classifies lines as left/right using horizontal position.
  - Calculates weight = distance from image center × number of detected lines.

- **Steering Angle Estimation**:
  - `(right_weight - left_weight) / (image_width / 2)` computes deviation.
  - Translates into a steering signal for the robot.

### Outputs
- `left_weight`, `right_weight`: Used in steering control and state vectors.
- `warped_image`, `cannyed_image`: Visuals for debugging.

---

## Configuration

- **Environment**:
  - `destination_coordinate`: goal position (default: `[-3.13, 0]`)
  - `reach_threshold`: distance tolerance to goal (default: `0.1`)
  - `safe_distance`: LiDAR-based obstacle distance (default: `0.35`)

- **Lane Detection**:
  - `center_margin`: default `50`, separates left/right lanes
  - Canny thresholds, color ranges tunable per environment

- **RL Hyperparameters**:
  - PPO: `learning_rate`, `clip_range`, `gae_lambda`, etc.
  - REINFORCE: `learning_rate`, `baseline`, `gamma`, etc.

---

## Logging & Visualization

- **TensorBoard**:
  ```bash
  tensorboard --logdir logs/train
  ```

- **WandB**: Logs metrics like `goal_count`, loss, gradients
- **Reward Plot**: `./results/reward_plot.png` (smoothed & raw)
- **Neural Network Visualization**: Interactive graph viewer

---

## Results & Progress

| Category         | Status                                       |
|------------------|----------------------------------------------|
| Navigation       | Agent reaches goal with stable lane control  |
| Vision           | Robust lane detection at 720×1280 resolution |
| Integration      | Working state vector from fused camera/LiDAR |
| Logging          | WandB + TensorBoard integrated               |
| Reward Function  | Penalizes collisions, rewards goal approach  |


