#!/usr/bin/env python3
"""
Simplified test script using only MuJoCo to verify robot model
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import op3_conf as conf

def test_mujoco_only():
    """Test MuJoCo simulation without TSID"""
    try:
        print("Loading MuJoCo model...")
        mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = conf.dt
        
        print(f"✓ Model loaded successfully")
        print(f"  - Number of joints: {mj_model.njnt}")
        print(f"  - Number of actuators: {mj_model.nu}")
        print(f"  - Number of bodies: {mj_model.nbody}")
        
        # Set initial pose (standing position)
        # The robot should start in a standing pose
        qpos = np.zeros(mj_model.nq)
        qpos[2] = 0.4  # Set initial height
        mj_data.qpos = qpos
        
        print("Starting MuJoCo viewer...")
        print("Press 'ESC' to exit")
        
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                t_elapsed = time.time() - start_time
                
                # Simple sinusoidal motion for testing
                if t_elapsed > 2.0:  # Wait 2 seconds before starting motion
                    # Apply simple sinusoidal motion to some joints
                    amplitude = 0.1
                    frequency = 1.0
                    
                    # Apply to hip joints (indices 8-12 for right leg, 13-17 for left leg)
                    if mj_model.nu >= 18:  # Check if we have enough actuators
                        mj_data.ctrl[8] = amplitude * np.sin(2 * np.pi * frequency * t_elapsed)  # Right hip pitch
                        mj_data.ctrl[13] = -amplitude * np.sin(2 * np.pi * frequency * t_elapsed)  # Left hip pitch
                
                # Step the simulation
                mujoco.mj_step(mj_model, mj_data)
                
                # Sync viewer
                viewer.sync()
                
                # Small delay to control simulation speed
                time.sleep(0.01)
        
        print("✓ MuJoCo simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ MuJoCo test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mujoco_only() 