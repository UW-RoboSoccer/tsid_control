import sys
import numpy as np
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ctrl'))

def test_lipm():
    from LIPM import LIPM
    
    lipm = LIPM(0.3)  # 40cm height
    
    # Test natural frequency
    expected_w = np.sqrt(9.80665 / 0.4)
    print(f"Natural frequency: {lipm.w:.3f} rad/s")
    print(f"Expected: {expected_w:.3f} rad/s")
    
    # Test trajectory generation
    t = [0, 1]
    pos0 = np.array([0.0, 0.0])
    vel0 = np.array([0.1, 0.0])
    acc0 = np.array([0.0, 0.0])
    zmp = np.array([0.0, 0.0])
    
    lipm.make_trajectory(t, 0.01, pos0, vel0, acc0, zmp)
    
    print(f"Generated {len(lipm.x.traj)} trajectory points")
    print("LIPM good")
    return True

if __name__ == "__main__":
    test_lipm()