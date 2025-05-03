import numpy as np
import matplotlib.pyplot as plt
import op3_conf as conf

v = 1.0
w = 0.0
T = 5.0
dt = 0.002

# initial position
x = 0.0
y = 0.0
z = 0.0
theta = 0.0

t = np.linspace(0, T, int(T / dt))
traj = np.zeros((len(t), 4))

for i in range(len(t)):
    x = x + v * dt * np.cos(theta)
    y = y + v * dt * np.sin(theta)
    theta = theta + w * dt
    traj[i, :] = [x, y, z, theta]

# Show the trajectory
plt.figure()
plt.plot(traj[:, 0], traj[:, 1])
plt.title('Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def gen_footsteps(traj, step_length, step_width):
    footsteps = []
    dist = 0
    right_foot = True

    for i in range(len(traj) - 1):
        dx = traj[i + 1, 0] - traj[i, 0]
        dy = traj[i + 1, 1] - traj[i, 1]

        dist += np.sqrt(dx ** 2 + dy ** 2)
        if dist >= step_length:
            tangent = [dx, dy]
            tangent /= np.linalg.norm(tangent)
            normal = [-tangent[1], tangent[0]]
            
            footstep = np.zeros((4,))
            footstep[0] = traj[i, 0] + step_width * normal[0] * (1 if right_foot else -1)
            footstep[1] = traj[i, 1] + step_width * normal[1] * (1 if right_foot else -1)
            footstep[2] = np.arctan2(tangent[1], tangent[0])
            footstep[3] = right_foot
            footsteps.append(footstep)
            dist = 0
            right_foot = not right_foot
    return footsteps

footsteps = gen_footsteps(traj, conf.step_length, conf.step_width)

# Show the footsteps
for footstep in footsteps:
    plt.plot(footstep[0], footstep[1], 'ro' if footstep[3] else 'bo')
plt.title('Footsteps')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def gen_footstep_traj(footsteps, step_height, step_time, step_length, dt):
    footstep_traj = []
    num_points = int(step_time / dt)
    t = np.linspace(0, 1, num_points)
    for i in range(len(footsteps) - 2):
        p0 = footsteps[i][:2]
        p1 = footsteps[i + 2][:2]

        x_traj = np.linspace(p0[0], p1[0], num_points)
        y_traj = np.linspace(p0[1], p1[1], num_points)
        z_traj = 4 * step_height * t * (1 - t)

        HTM = []
        yaw = footsteps[i + 1][2]
        for j in range(num_points):
            T = np.eye(4)
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
            T[:3, :3] = R
            T[:3, 3] = [x_traj[j], y_traj[j], z_traj[j]]
            HTM.append(T)

        footstep_traj.append(HTM)
    return footstep_traj

# Show the footstep trajectory in 3D
footstep_traj = gen_footstep_traj(footsteps, conf.step_height, conf.step_time, conf.step_length, dt)
fig = plt.figure()
ax = plt.axes(projection='3d')
for traj in footstep_traj:
    x = [step_traj[0, 3] for step_traj in traj]
    y = [step_traj[1, 3] for step_traj in traj]
    z = [step_traj[2, 3] for step_traj in traj]
    ax.plot3D(x, y, z, 'b-')
ax.set_title('Footstep Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(0, 1)
ax.set_ylim(-1, 1)
plt.show()


