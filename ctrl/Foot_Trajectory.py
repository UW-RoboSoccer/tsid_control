import numpy as np
from typing import List, Tuple
from scipy.interpolate import CubicSpline

class FootTrajectory:
    def __init__(self, t: List, start: np.ndarray, target: np.ndarray, step_height: float, rise_ratio: float=0.5):
        self.t = t
        self.x = CubicSpline(t, [start[0], target[0]])
        self.y = CubicSpline(t, [start[1], target[1]])
        self.z = None
        self.yaw = CubicSpline(t, [start[3], target[3]]) if len(start) > 3 else None
        
        duration = t[1] - t[0]
        if rise_ratio != 0.5:
            rise_time = duration * rise_ratio
            new_t = [t[0], t[0] + rise_time, t[1] - rise_time, t[1]]
            self.z = CubicSpline(new_t, [start[2], start[2] + step_height, target[2] + step_height, target[2]])
        else:
            self.z = CubicSpline([t[0], t[0] + duration * rise_ratio, t[1]], [start[2], start[2] + step_height, target[2]])

    def get_position(self, t: float) -> np.ndarray:
        """
        Get the foot position at time t.
        :param t: Time at which to get the position.
        :return: Foot position as a numpy array.
        """
        return np.array([self.x(t), self.y(t), self.z(t)])
    
    def get_velocity(self, t: float) -> np.ndarray:
        """
        Get the foot velocity at time t.
        :param t: Time at which to get the velocity.
        :return: Foot velocity as a numpy array.
        """
        return np.array([self.x(t, 2), self.y(t, 2), self.z(t, 2)])
    
    def get_acceleration(self, t: float) -> np.ndarray:
        """
        Get the foot acceleration at time t.
        :param t: Time at which to get the acceleration.
        :return: Foot acceleration as a numpy array.
        """
        return np.array([self.x(t, 3), self.y(t, 3), self.z(t, 3)])
    
if __name__ == "__main__":
    t = [0, 1]  # Start and end times
    start = np.array([0, 0, 0])  # Start position
    target = np.array([1, 1, 0])  # Target position
    step_height = 0.2  # Height of the step
    trajectory = FootTrajectory(t, start, target, step_height)

    # Plot position in 3D
    import matplotlib.pyplot as plt
    x = trajectory.x(np.linspace(t[0], t[1], 100))
    y = trajectory.y(np.linspace(t[0], t[1], 100))
    z = trajectory.z(np.linspace(t[0], t[1], 100))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.title('Foot Trajectory')
    plt.show()

    rise_ratio = 0.1
    trajectory = FootTrajectory(t, start, target, step_height, rise_ratio)
    # Plot position in 3D with different rise ratio
    x = trajectory.x(np.linspace(t[0], t[1], 100))
    y = trajectory.y(np.linspace(t[0], t[1], 100))
    z = trajectory.z(np.linspace(t[0], t[1], 100))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.title('Foot Trajectory with Rise Ratio {}'.format(rise_ratio))
    plt.show()