import numpy as np


class TrajectoryGenerator:
    @staticmethod
    def generate_circle_trajectory(total_time, dt, radius=1.0, omega=1.0):
        """
        Generates circular trajectory: [x, y, theta]
        """
        t = np.arange(0, total_time, dt)
        x_ref = radius * np.cos(omega * t)
        y_ref = radius * np.sin(omega * t)
        theta_ref = omega * t + (np.pi / 2)
        theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi

        return np.column_stack((x_ref, y_ref, theta_ref))

    @staticmethod
    def generate_sin_cos_trajectory(total_time, dt, x0=None):
        """
        Generates trajectory controlled by u0=1, u1=cos(t)
        Returns: N x 3 array [x, y, theta]
        """
        # Time discretization
        t = np.arange(0, total_time, dt)
        N = t.shape[0]

        # Prepare state arrays
        x = np.zeros(N)
        y = np.zeros(N)
        theta = np.zeros(N)

        # Set initial state
        if x0 is not None:
            x[0], y[0], theta[0] = x0
        # Default is already [0,0,0]

        # Euler Integration
        for i in range(1, N):
            ti = t[i - 1]
            # Inputs
            u0 = 1
            u1 = np.cos(ti)

            # State derivatives
            dx_val = u0 * np.cos(theta[i - 1])
            dy_val = u0 * np.sin(theta[i - 1])
            dtheta_val = u1

            # Update
            x[i] = x[i - 1] + dx_val * dt
            y[i] = y[i - 1] + dy_val * dt
            theta[i] = theta[i - 1] + dtheta_val * dt

        return np.column_stack((x, y, theta))
