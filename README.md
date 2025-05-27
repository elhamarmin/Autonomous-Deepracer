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
  Seamless transfer to AWS DeepRacer's camera, IMU, and motor interface.

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

The project provides a simple interface through the `rl_gym.py` script in the `WebotsController` folder. Users only need to modify the `run_model` variable to switch between different modes:

### Running the Project

Navigate to the `WebotsController` folder and modify the `run_model` variable in `rl_gym.py`:

```python
run_model = 0  # 0 = Train, 1 = Test, 2 = Debug
```

Then run:
```bash
cd WebotsController
python rl_gym.py
```

### Run Modes

#### Mode 0: Training
```python
run_model = 0  # Training mode
```
- Automatically starts training using Weights & Biases sweeps
- Uses PPO algorithm with hyperparameter optimization
- Models are saved to `./trained_model/model_<timestamp>.zip`
- For manual training, uncomment and modify the `manual_train_model` call

#### Mode 1: Testing
```python
run_model = 1  # Testing mode
```
- Loads a pre-trained model and runs inference
- Update the `model_path` variable to point to your trained model:
  ```python
  model_path = './trained_model/model_20240723-1425.zip'
  ```
- Displays real-time forward and steering values

#### Mode 2: Debug/Manual Control
```python
run_model = 2  # Debug mode
```
- Allows manual control of the robot using keyboard inputs:
  - `w`: Move forward
  - `s`: Move backward  
  - `a`: Turn left
  - `d`: Turn right
  - `i`: Step simulation
- Displays LiDAR sensor data for debugging

### Custom OpenAI Gym Environment

The project includes a custom OpenAI Gym environment (`WebotsGymEnv`) that wraps the Webots simulation:

- **Environment Class**: `WebotsGymEnv` in `WebotsGymEnv.py`
- **Base Environment**: `WebotsEnvironment` in `WebotsEnvironment.py`
- **Action Space**: Continuous 2D space `[-1,1]` for [forward/backward, left/right]
- **Observation Space**: Camera-based lane detection weights and line counts

#### Customizing the Environment

Users can modify the environment to meet their specific needs:

1. **Reward Function**: Modify `get_reward()` in `WebotsEnvironment.py`
2. **Observation Space**: Adjust `get_observations()` for different sensor inputs
3. **Action Space**: Change action dimensions in `WebotsGymEnv.py`
4. **Environment Parameters**: 
   - `destination_coordinate`: Goal position (default: `[-3.13, 0]`)
   - `reach_threshold`: Distance threshold for goal (default: `0.1`)
   - `safe_distance`: Obstacle avoidance distance (default: `0.35`)
   - `max_speed`: Maximum robot speed (default: `20`)

#### Example Environment Usage
```python
from WebotsGymEnv import WebotsGymEnv

env = WebotsGymEnv()
obs = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs = env.reset()
```

### Alternative Training Methods

For advanced users who want more control over the training process:

#### Manual PPO Training
```python
from MacRacerTraining import MacRacerTraining

macRacerTr = MacRacerTraining(train=True)
macRacerTr.manual_train_model(
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

#### REINFORCE Training
```python
from your_script import Agent_REINFORCE, Policy_Network, WebotsGymEnv
import torch

env = WebotsGymEnv()
policy = Policy_Network(3, 6, 3).to(device)
agent = Agent_REINFORCE(
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

### Monitoring and Visualization

- **TensorBoard**: View training logs
  ```bash
  tensorboard --logdir logs/train
  ```
- **Weights & Biases**: Automatic experiment tracking and hyperparameter sweeps
- **Real-time Output**: Forward and steering values displayed during testing

---

## Lane Detection System

Pipeline steps:

1. **ROI Crop**  
   Trapezoid mask via `region_of_interest`.
2. **Color Filter**  
   Yellow range in BGR → grayscale.
3. **Perspective Warp**  
   Bird's-eye via `perspective_transform`.
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
   - Match DeepRacer's camera (1080×720).  
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

