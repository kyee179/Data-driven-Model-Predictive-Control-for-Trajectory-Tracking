import numpy as np


class TrajectoryGenerator:
    """
    Module for generating reference trajectories for the robot to follow.
    """

    @staticmethod
    def generate_circle(
        total_time: float, dt: float, radius: float = 1.0, omega: float = 1.0
    ) -> np.ndarray:
        """
        Generates a circular reference trajectory.

        Args:
            total_time (float): Total duration of the trajectory.
            dt (float): Time step duration.
            radius (float): Radius of the circle path.
            omega (float): Angular velocity (speed of rotation).

        Returns:
            np.ndarray: A matrix of shape (N, 3) where each row is [x, y, theta].
        """
        t = np.arange(0, total_time, dt)
        x_ref = radius * np.cos(omega * t)
        y_ref = radius * np.sin(omega * t)
        theta_ref = omega * t
        return np.column_stack((x_ref, y_ref, theta_ref))
