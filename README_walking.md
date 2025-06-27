# Bipedal Walking Simulation

This directory contains a complete bipedal walking simulation system using MuJoCo and TSID (Task Space Inverse Dynamics) control.

## Files Overview

### Main Simulation Files
- **`walking_simulation.py`** - Complete walking simulation with MuJoCo visualization
- **`test_walking.py`** - Test script to verify trajectory generation without MuJoCo
- **`walk_controller.py`** - Walking controller using DCM (Divergent Component of Motion) approach
- **`biped_balance.py`** - Original balance simulation (reference implementation)

### Supporting Files
- **`biped.py`** - Biped robot model with TSID formulation
- **`op3_conf.py`** - Configuration parameters
- **`robot/`** - Robot model files (URDF, SRDF, MuJoCo XML)

## Quick Start

### 1. Test the Walking Controller (Recommended First Step)
```bash
python test_walking.py
```
This will test the trajectory generation and show plots without running MuJoCo simulation.

### 2. Run the Full Walking Simulation
```bash
python walking_simulation.py
```
This will open a MuJoCo viewer showing the biped robot walking.

## Features

### Walking Controller (`walk_controller.py`)
- **DCM-based walking**: Uses Divergent Component of Motion for stability
- **Footstep planning**: Generates alternating left/right footsteps
- **ZMP trajectory**: Zero Moment Point reference generation
- **CoM control**: Center of Mass trajectory tracking

### Simulation Features
- **MuJoCo integration**: Real-time physics simulation
- **Visual markers**: 
  - Red sphere: CoM reference
  - Green sphere: Current CoM
  - Cyan sphere: Left foot
  - Blue sphere: Right foot
  - Yellow cylinder: Capture point
- **Trajectory plotting**: Post-simulation analysis plots

## Configuration

Key parameters in `op3_conf.py`:
```python
step_length = 0.1      # Length of each step (m)
step_height = 0.05     # Height of foot swing (m)
step_width = 0.1275    # Width between feet (m)
step_time = 0.7        # Time per step (s)
dt = 0.002            # Simulation time step (s)
```

## Customization

### Modify Walking Path
In `walking_simulation.py`, change the path generation:
```python
# For straight walking
traj = generate_walking_path(linear_vel=0.3, angular_vel=0.0, duration=8.0)

# For curved walking
traj = generate_walking_path(linear_vel=0.3, angular_vel=0.2, duration=8.0)
```

### Adjust Controller Parameters
Modify the DCM controller gains in `walk_controller.py`:
```python
# In dcm_controller method
vrp_control = vrp_i + (1 + self.conf.k_dcm * self.w_n) * (self.e - dcm_ref)
```

## Troubleshooting

### Common Issues

1. **MuJoCo model not found**
   - Ensure `robot/v1/mujoco/robot.xml` exists
   - Check path in `op3_conf.py`

2. **TSID solver fails**
   - Check joint limits and configuration
   - Verify contact constraints

3. **Walking instability**
   - Reduce step length or increase step time
   - Adjust DCM controller gains
   - Check CoM height configuration

### Debug Mode
Run the test script first to verify trajectory generation:
```bash
python test_walking.py
```

## Dependencies

Required Python packages:
- `mujoco`
- `numpy`
- `matplotlib`
- `scipy`
- `tsid` (Task Space Inverse Dynamics)
- `pinocchio` (robot modeling)

## References

- DCM-based walking control
- TSID for whole-body control
- MuJoCo physics simulation
- Capture point theory for bipedal stability 