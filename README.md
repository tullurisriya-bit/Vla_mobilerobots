# OmniVLA Mobile Robot Navigation using ROS2

## Overview

This repository contains the ROS2 integration of **OmniVLA** for autonomous mobile robot navigation using a **WHILL** mobile robot.

The project combines the OmniVLA vision-language-action model with ROS2 to perform real-time navigation by subscribing to camera images, running OmniVLA inference, and publishing velocity commands to the robot.

The primary contribution of this repository is the **`omnivla_ros`** package, which integrates OmniVLA into a ROS2 ecosystem.

---

# Repository Structure

```text
Project_Files_GitHub/
├── README.md
├── .gitignore
└── src/
    ├── OmniVLA/
    ├── omnivla_ros/
    ├── ros2_whill/
    └── ros2_whill_interfaces/
```

## Package Description

### OmniVLA

Contains the original OmniVLA source code used for visual navigation and policy inference.

Repository:
https://github.com/NHirose/OmniVLA

---

### omnivla_ros

This package provides the ROS2 interface for OmniVLA.

Responsibilities include:

* Loading the OmniVLA model
* Receiving camera images
* Preprocessing observations
* Running model inference
* Converting predicted actions into ROS2 messages
* Publishing robot velocity commands
* Interfacing OmniVLA with the WHILL robot

This package contains the main contribution of this project.

---

### ros2_whill

ROS2 driver for communicating with the WHILL mobile robot.

---

### ros2_whill_interfaces

Contains custom ROS2 message and service definitions required by the WHILL driver.

---

# System Architecture

```
Camera
   │
   ▼
Image Topic
   │
   ▼
omnivla_ros
   │
   ├── Load OmniVLA
   ├── Run inference
   ├── Predict robot action
   ▼
cmd_vel
   │
   ▼
WHILL Driver
   │
   ▼
WHILL Robot
```

---

# Requirements

* Ubuntu 22.04
* ROS2 Humble
* Python 3.10
* Conda
* CUDA compatible GPU (recommended)

---

# Workspace Layout

The ROS2 workspace should have the following structure:

```text
ros2_ws/
├── src/
│   ├── OmniVLA/
│   ├── omnivla_ros/
│   ├── ros2_whill/
│   └── ros2_whill_interfaces/
├── build/
├── install/
└── log/
```

---

# Build

```bash
cd ~/ros2_ws

colcon build

source install/setup.bash
```

---

# Running

Launch the required ROS2 nodes and then start the OmniVLA node.

```bash
ros2 run omnivla_ros omnivla_node
```

---

# OmniVLA Model Checkpoints

The trained OmniVLA model weights are **NOT included** in this repository because they exceed GitHub's file size limits.

After cloning the project, download the checkpoints separately by following the official OmniVLA instructions.

Place the downloaded checkpoint folders inside:

```text
src/OmniVLA/
├── omnivla-original/
├── omnivla-original-balance/
└── omnivla-finetuned-cast/
```

These folders should contain the downloaded `.safetensors`, `.pt`, and related checkpoint files required for inference.

---

# Notes

* This repository contains the ROS2 integration code for OmniVLA.
* The original OmniVLA implementation is developed by the OmniVLA authors.
* The downloaded model checkpoints are intentionally excluded from version control.

---

# Future Work

* Goal image navigation
* Language-conditioned navigation
* Satellite-image navigation
* Multi-modal goal conditioning
* Real-world WHILL deployment
* Additional ROS2 launch files
* Simulation support

---

# License

Please refer to the licenses of the original OmniVLA and WHILL repositories for their respective source code.

The ROS2 integration code (`omnivla_ros`) follows the license provided in this repository.

