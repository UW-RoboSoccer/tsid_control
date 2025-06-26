#!/usr/bin/env python3
"""
Simple test script to verify TSID setup works
"""

import numpy as np
import op3_conf as conf
from biped import Biped

def test_tsid_setup():
    """Test if TSID setup works correctly"""
    try:
        print("Initializing biped robot...")
        biped = Biped(conf)
        print("✓ Biped initialization successful")
        
        print("Testing TSID formulation...")
        q, v = biped.q, biped.v
        HQPData = biped.formulation.computeProblemData(0.0, q, v)
        print("✓ TSID formulation successful")
        
        print("Testing QP solver...")
        sol = biped.solver.solve(HQPData)
        if sol.status == 0:
            print("✓ QP solver successful")
        else:
            print(f"✗ QP solver failed with status: {sol.status}")
            
        print("Testing robot state...")
        com = biped.robot.com(biped.formulation.data())
        print(f"✓ Center of mass: {com}")
        
        print("Testing capture point...")
        com_vel = biped.robot.com_vel(biped.formulation.data())
        w = np.sqrt(9.81 / com[2])
        cp = biped.compute_capture_point(com, com_vel, w)
        print(f"✓ Capture point: {cp}")
        
        print("\n🎉 All tests passed! TSID setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tsid_setup() 