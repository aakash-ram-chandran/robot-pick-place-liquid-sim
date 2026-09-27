import math

import numpy as np


def robot_to_world_rotation():
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def to_world(T):
    C = robot_to_world_rotation()
    W = np.eye(4)
    W[:3, :3] = C @ T[:3, :3]
    W[:3, 3] = C @ T[:3, 3]
    return W


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rot_axis(axis, a):
    k = np.asarray(axis, float)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def quat_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def quat_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def blend(Ta, Tb, s):
    qa, qb = np.array(quat_wxyz(Ta[:3, :3])), np.array(quat_wxyz(Tb[:3, :3]))
    if qa @ qb < 0.0:
        qb = -qb
    ang = math.acos(min(1.0, float(qa @ qb)))
    q = qa if ang < 1e-6 else (math.sin((1 - s) * ang) * qa + math.sin(s * ang) * qb) / math.sin(ang)
    return pose(quat_matrix(*(q / np.linalg.norm(q))), (1 - s) * Ta[:3, 3] + s * Tb[:3, 3])


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def set_pose(obj, T):
    obj.position.set(*T[:3, 3])
    w, qx, qy, qz = quat_wxyz(T[:3, :3])
    obj.quaternion.set(qx, qy, qz, w)


def body_pose(body):
    p, q = body.position, body.quaternion
    return pose(quat_matrix(q.w, q.x, q.y, q.z), (p.x, p.y, p.z))
