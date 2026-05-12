# YOPO LiDAR + ROS2 — Build and run

This repository is **LiDAR-only** and **ROS2-only**: depth-image modes, `sensor_mode` toggles, and ROS1 entry points were removed.

## 1. Requirements

- Ubuntu 22.04 (or compatible) with **ROS2 Humble**
- NVIDIA driver + CUDA (for `sensor_simulator`)
- Conda with **Python 3.10** for `rclpy` compatibility

## 2. Clone layout

Assume the workspace root is `YOPO_lidar/` with subfolders `YOPO/`, `Controller/`, `Simulator/`.

## 3. Python environment

```bash
conda create -n yopo python=3.10 -y
conda activate yopo
cd YOPO_lidar/YOPO
pip install -r requirements.txt
```

## 4. Build ROS2 packages

```bash
cd YOPO_lidar/YOPO
source /opt/ros/humble/setup.bash
bash scripts/build_controller_ros2.sh
bash scripts/build_simulator_ros2.sh
```

Artifacts:

- `Controller/src/install_ros2/`
- `Simulator/src/install_ros2/`

Optional: `export YOPO_CUDA_ARCH=86` (or your SM) if `nvcc` arch detection fails.

## 5. Full stack (single launch)

```bash
cd YOPO_lidar/YOPO
source /opt/ros/humble/setup.bash
source ../Controller/src/install_ros2/setup.bash
source ../Simulator/src/install_ros2/setup.bash
ros2 launch yopo_bringup system.launch.py weight:=saved/YOPO_1/epoch50.pth
```

(`conda run -n yopo` is wired inside this launch file for the planner.)

## 6. Manual three-terminal flow

### Terminal A — quadrotor + attitude control

```bash
cd YOPO_lidar
source /opt/ros/humble/setup.bash
source Controller/src/install_ros2/setup.bash
ros2 launch so3_quadrotor_simulator simulator_attitude_control.launch.py
```

### Terminal B — LiDAR simulator

```bash
cd YOPO_lidar
source /opt/ros/humble/setup.bash
source Controller/src/install_ros2/setup.bash
source Simulator/src/install_ros2/setup.bash
ros2 run sensor_simulator sensor_simulator
```

### Terminal C — YOPO planner

```bash
cd YOPO_lidar
source /opt/ros/humble/setup.bash
source Controller/src/install_ros2/setup.bash
conda activate yopo
cd YOPO
python test_yopo_ros.py --ros_version ros2 --sensor_mode lidar --weight saved/YOPO_2/epoch1.pth
```

### Terminal D — RViz (optional)

```bash
cd YOPO_lidar/YOPO
rviz2 -d yopo_ros2.rviz
```

## 7. Health checks (topics)

```bash
source /opt/ros/humble/setup.bash
source YOPO_lidar/Controller/src/install_ros2/setup.bash
source YOPO_lidar/Simulator/src/install_ros2/setup.bash
ros2 topic echo /sim/odom --once
ros2 topic echo /lidar_points --once
ros2 topic echo /so3_control/pos_cmd --once
```

## 8. Dataset collection (`lidar_*.bin` only)

```bash
source /opt/ros/humble/setup.bash
source YOPO_lidar/Simulator/src/install_ros2/setup.bash
ros2 run sensor_simulator dataset_generator
```

Outputs under `Simulator/src/config/config.yaml` → `save_path` (default `../dataset/`): folders `0/`, `1/`, … with `lidar_i.bin` plus `pose-*.csv` at dataset root.

Keep **`Simulator` LiDAR layout** aligned with **`YOPO/config/traj_opt.yaml`** (`range_image_height` / `range_image_width` / `lidar_vertical_fov` / `lidar_sensing_horizon`).

## 9. Training

```bash
conda activate yopo
cd YOPO_lidar/YOPO
python train_yopo.py
```

Training loads only `*.bin` + `pose-*.csv` (see `policy/yopo_dataset.py`).

## 10. Inference CLI

```bash
python test_yopo_ros.py --ros_version ros2 --weight saved/YOPO_1/epoch50.pth \
  --odom_topic /sim/odom --lidar_topic /lidar_points --ctrl_topic /so3_control/pos_cmd
```

## 11. Configuration reference

| File | Role |
|------|------|
| `Simulator/src/config/config.yaml` | Map generation, `lidar:` ray grid, `lidar_fps`, dataset paths |
| `YOPO/config/traj_opt.yaml` | Range image size, lattice / primitives, dataset_path, speeds |

`YOPO/config/config.py` always applies the **360° LiDAR lattice** derivations (no `sensor_mode`).

## 12. Launch arguments (`yopo_bringup`)

- `yopo_root` — absolute path to the `YOPO/` directory (defaults beside `Controller/` in this repo)
- `weight` — `.pth` file relative to `yopo_root` unless absolute

Example:

```bash
ros2 launch yopo_bringup full_yopo_ros2.launch.py \
  yopo_root:=/home/you/YOPO_lidar/YOPO \
  weight:=saved/YOPO_1/epoch50.pth
```

## 13. Troubleshooting

- **`ModuleNotFoundError: rclpy`** — use Python **3.10** in the environment that runs `test_yopo_ros.py`.
- **`quadrotor_msgs` not found** — source `Controller/src/install_ros2/setup.bash` before Python.
- **Empty dataset / wrong path** — ensure `dataset_path` in `traj_opt.yaml` points to the folder that contains `pose-0.csv` and map subfolders (paths are resolved from `YOPO/`).

## 14. Migration note

Old commands (`rosrun`, `catkin_make`, `--trial/--epoch`, `--sensor_mode depth`, `/depth_image`) are intentionally unsupported; use the ROS2 + LiDAR flow above.
