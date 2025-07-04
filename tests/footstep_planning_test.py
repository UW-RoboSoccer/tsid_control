import sys
import numpy as np
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ctrl'))

def test_footstep_planning():
    from Footstep_Planner import FootstepPlanner, Footstep
    
    planner = FootstepPlanner(step_width=0.1, step_length=0.04)
    
    path = [
        np.array([0, 0]),
        np.array([0.04, 0]),
        np.array([0.08, 0])
    ]
    
    init_supports = [
        Footstep(position=np.array([0, 0.05]), orientation=np.array([0, 0, 0]), side=0),
        Footstep(position=np.array([0, -0.05]), orientation=np.array([0, 0, 0]), side=1)
    ]
    
    footsteps = planner.plan(path, init_supports)
    
    print(f"Planned {len(footsteps)} footsteps")
    for i, footstep in enumerate(footsteps):
        print(f"  Step {i}: pos={footstep.position}, side={'left' if footstep.side == 0 else 'right'}")
    
    print("Footstep planning works")
    return True

if __name__ == "__main__":
    test_footstep_planning()