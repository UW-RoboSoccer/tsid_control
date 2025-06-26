# TSID Control System Setup Guide

This guide will help you set up and run the TSID (Task Space Inverse Dynamics) bipedal robot control system in MuJoCo.

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. Required Dependencies

#### Core Dependencies (Install via pip):
```bash
pip install numpy scipy matplotlib mujoco
```

#### Advanced Dependencies (May require special installation):

**Pinocchio:**
- **Option 1 (Recommended):** Install via conda
  ```bash
  conda install -c conda-forge pinocchio
  ```
- **Option 2:** Install from source
  ```bash
  git clone https://github.com/stack-of-tasks/pinocchio.git
  cd pinocchio
  mkdir build && cd build
  cmake ..
  make -j4
  sudo make install
  ```

**TSID:**
- **Option 1 (Recommended):** Install via conda
  ```bash
  conda install -c conda-forge tsid
  ```
- **Option 2:** Install from source
  ```bash
  git clone https://github.com/stack-of-tasks/tsid.git
  cd tsid
  mkdir build && cd build
  cmake ..
  make -j4
  sudo make install
  ```

## Installation Steps

### Step 1: Install Basic Dependencies
```bash
pip install numpy scipy matplotlib mujoco
```

### Step 2: Install Pinocchio and TSID
```bash
# Using conda (recommended)
conda install -c conda-forge pinocchio tsid

# Or install manually if conda is not available
```

### Step 3: Verify Installation
```bash
python test_simple.py
```

### Step 4: Run the Full Simulation
```bash
python biped_balance.py
```

## Troubleshooting

### Common Issues:

1. **Pinocchio/TSID Import Errors:**
   - These libraries are complex and may require specific installation methods
   - Try using conda instead of pip
   - Check if you need to set LD_LIBRARY_PATH

2. **MuJoCo Model Path Issues:**
   - Ensure the robot XML file exists at `./robot/robot.xml`
   - Check that all mesh files are in the correct location

3. **QP Solver Issues:**
   - The system uses HQuadProg solver which should be included with TSID
   - If solver fails, check if TSID is properly installed

### Alternative Setup (Using Docker):

If you're having issues with dependencies, you can use a pre-configured Docker environment:

```bash
# Pull a robotics Docker image with Pinocchio and TSID
docker pull stackoftasks/pinocchio:latest

# Run the container
docker run -it --rm -v $(pwd):/workspace stackoftasks/pinocchio:latest
```

## Running the Simulation

### Basic Balance Control:
```bash
python biped_balance.py
```

This will:
- Initialize the biped robot in MuJoCo
- Start a balance control simulation
- Show real-time visualization with:
  - Red sphere: CoM reference
  - Green sphere: Actual CoM
  - Cyan/Blue spheres: Foot positions
  - Yellow cylinder: Capture point

### Walking Controller:
```bash
python walk_controller.py
```

### Test Individual Components:
```bash
python test_capture_point.py  # Test capture point computation
python gen_footsteps.py       # Test footstep generation
```

## Configuration

The main configuration is in `op3_conf.py`:
- `step_length`, `step_height`, `step_width`: Walking parameters
- `w_com`, `w_foot`, `w_posture`: Task weights
- `kp_*`: Control gains
- `dt`: Time step (0.002s = 500Hz)

## Expected Output

When running successfully, you should see:
1. A MuJoCo viewer window with the robot
2. Real-time visualization of robot state
3. Console output showing control status
4. After 5 seconds, a simulated push disturbance

## Notes

- The system runs at 500Hz control frequency
- The robot model is based on the OP3 humanoid
- The control uses TSID for whole-body control
- DCM (Divergent Component of Motion) is used for walking stability 