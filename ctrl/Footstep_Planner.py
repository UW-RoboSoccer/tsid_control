import numpy as np
from typing import List, Tuple

class Footstep:
    def __init__(self, position: np.ndarray, orientation: np.ndarray, side: int):
        self.frame = np.eye(3)  # 3x3 transformation matrix
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.side = side # 0 for left, 1 for right foot

        self.update_frame()

    def update_frame(self):
        # Update the transformation matrix based on position and orientation
        self.frame[:2, 2] = self.position
        self.frame[:2, :2] = self.rotation_matrix(self.orientation)

    def rotation_matrix(self, orientation: float) -> np.ndarray:
        # Assuming an orientation about the z axis (yaw) in 2D
        sz, cz = np.sin(orientation[2]), np.cos(orientation[2])

        R = np.array(
            [[cz, -sz],
             [sz, cz]]
        )

        return R
    
    def transform(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a point using the footstep's frame.
        :param point: A 2D point to be transformed.
        :return: Transformed point in the footstep's coordinate system.
        """
        return self.frame[:2, :2] @ point + self.frame[:2, 2]

    def __repr__(self):
        return f"Footstep(position={self.position}, orientation={self.orientation})"
    
class Support:
    def __init__(self, contacts: List[Footstep], foot_width: float, foot_length: float, start_time: float = 0.0):
        self.contacts = contacts
        self.is_double_support = len(contacts) == 2
        self.foot_width = foot_width
        self.foot_length = foot_length
        self.start_time = start_time

    def get_support_polygon(self) -> List[np.ndarray]:
        """
        Get the support polygon formed by the footstep contact(s)
        :return: Vertices of the support polygon in clockwise order.
        """
        polygon = []
        for contact in self.contacts:
            if contact.side == 0:  # Left foot
                polygon.append(contact.transform([-self.foot_length / 2, self.foot_width / 2]))
                polygon.append(contact.transform([-self.foot_length / 2, -self.foot_width / 2]))
                polygon.append(contact.transform([self.foot_length / 2, -self.foot_width / 2]))
                polygon.append(contact.transform([self.foot_length / 2, self.foot_width / 2]))
            else:  # Right foot
                polygon.append(contact.transform([self.foot_length / 2, -self.foot_width / 2]))
                polygon.append(contact.transform([self.foot_length / 2, self.foot_width / 2]))
                polygon.append(contact.transform([-self.foot_length / 2, self.foot_width / 2]))
                polygon.append(contact.transform([-self.foot_length / 2, -self.foot_width / 2]))

        return polygon


class FootstepPlanner:
    def __init__(self, step_width, step_length):
        self.step_width = step_width
        self.step_length = step_length

    def add_step(self, dx, dy, side: int, pos: np.ndarray) -> Footstep:
        """
        Create a footstep based on the displacement and side.
        :param dx: Displacement in x direction.
        :param dy: Displacement in y direction.
        :param side: 0 for left foot, 1 for right foot.
        :param pos: Current position of the robot.
        :return: Footstep object.
        """
        tangent = np.array([dx, dy])
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])

        position = pos + tangent * (self.step_length / 2) + normal * (self.step_width / 2 * (1 if side == 0 else -1))
        orientation = np.array([0, 0, np.arctan2(dy, dx)])

        return Footstep(position=position, orientation=orientation, side=side)

    def plan(self, path: List[np.ndarray], init_supports: List[Footstep]) -> List[Footstep]:
        """
        Plan footsteps along a given path.
        :param path: List of positions [x, y] to follow.
        :return: List of Footstep objects.
        """
        footsteps = []
        footsteps.extend(init_supports)
        side = init_supports[-1].side
        distance = 0.0

        for i in range(len(path) - 1):
            dx, dy = path[i + 1] - path[i]
            distance += np.linalg.norm([dx, dy])

            if distance >= self.step_length:
                side = not side  # Switch foot
                footstep = self.add_step(dx, dy, side, path[i])

                footsteps.append(footstep)
                distance = 0.0

        # Ensure the last footstep is added
        side = not side
        footstep = self.add_step(dx, dy, side, path[-1])
        footsteps.append(footstep)

        # Ensure the path ends with double support if the last segment is not enough for a full step
        if distance > 0:
            side = not side
            footstep = self.add_step(dx, dy, side, path[-1])
            footsteps.append(footstep)

        return footsteps

# Example usage:
planner = FootstepPlanner(step_width=0.2, step_length=0.3)

## Generate a simple path
dt = 0.1
t = np.linspace(0, 10, 100)
v, w = 0.5, 0.1  # linear and angular velocity
x, y = 0, 0  # initial position
theta = 0  # initial orientation
path = []
for i in range(len(t)):
    x += v * dt * np.cos(theta)
    y += v * dt * np.sin(theta)
    theta += w * dt
    path.append(np.array([x, y]))

# Initial supports (left and right foot)
init_supports = [
    Footstep(position=np.array([0, 0.1]), orientation=np.array([0, 0, 0]), side=0),  # Left foot
    Footstep(position=np.array([0, -0.1]), orientation=np.array([0, 0, 0]), side=1)   # Right foot
]

# Plan footsteps along the path
footsteps = planner.plan(path, init_supports)

# Plot the footsteps
import matplotlib.pyplot as plt
path = np.array(path)
plt.figure(figsize=(10, 5))
# Define footstep dimensions
foot_length = 0.25  # length of the foot in meters
foot_width = 0.1    # width of the foot in meters

for footstep in footsteps:
    # Create rectangle centered at footstep position with proper orientation
    angle_rad = footstep.orientation[2]
    
    # Create a rotated rectangle using matplotlib patches
    rect = plt.Rectangle(
        footstep.transform([-foot_length/2, -foot_width/2]),  # centered at origin initially
        foot_length, foot_width,
        angle=np.degrees(angle_rad),
        color='r' if footstep.side == 0 else 'b',
        alpha=0.7,
    )
    plt.gca().add_patch(rect)
    
    # Add a small dot at the center for reference
    plt.plot(footstep.position[0], footstep.position[1], 'ko', markersize=2)
    
# Plot the path
plt.plot(path[:, 0], path[:, 1], 'g--', label='Planned Path')

plt.legend()
plt.title('Planned Footsteps')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.grid()
plt.show()