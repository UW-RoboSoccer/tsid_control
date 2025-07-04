# test_03_constants.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ctrl'))


def test_constants():
    from const import dt, step_height, step_width, step_length, step_duration
    
    print(f"dt: {dt}s")
    print(f"step_height: {step_height}m")
    print(f"step_width: {step_width}m")
    print(f"step_length: {step_length}m")
    print(f"step_duration: {step_duration}s")
    
    # Verify reasonable ranges
    assert 0.001 <= dt <= 0.01, "dt should be 1-10ms"
    assert 0.005 <= step_height <= 0.05, "step_height should be 5-50mm"
    assert 0.05 <= step_width <= 0.2, "step_width should be 5-20cm"
    assert 0.02 <= step_length <= 0.1, "step_length should be 2-10cm"
    
    print("constants are reasonable")
    return True

if __name__ == "__main__":
    test_constants()