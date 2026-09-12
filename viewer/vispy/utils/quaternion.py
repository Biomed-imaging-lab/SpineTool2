import numpy as np


def quaternion2euler(quaternion, degrees=False):
    epsilon = 1e-10

    q = quaternion

    sin_theta_2 = 2 * (q.w * q.y - q.z * q.x)
    sin_theta_2 = np.sign(sin_theta_2) * min(abs(sin_theta_2), 1)

    if abs(sin_theta_2) > 1 - epsilon:
        theta_1 = -np.sign(sin_theta_2) * 2 * np.arctan2(q.x, q.w)
        theta_2 = np.arcsin(sin_theta_2)
        theta_3 = 0

    else:
        theta_1 = np.arctan2(
            2 * (q.w * q.z + q.y * q.x),
            1 - 2 * (q.y * q.y + q.z * q.z),
        )

        theta_2 = np.arcsin(sin_theta_2)

        theta_3 = np.arctan2(
            2 * (q.w * q.x + q.y * q.z),
            1 - 2 * (q.x * q.x + q.y * q.y),
        )

    angles = (theta_1, theta_2, theta_3)

    if degrees:
        return tuple(np.degrees(angles))

    return angles
