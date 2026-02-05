import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    @staticmethod
    def plot_tracking(ref_traj, x_history, u_history, dt):
        """
        Plot tracking performance and control inputs.
        """
        x_history = np.array(x_history)
        u_history = np.array(u_history)
        time_steps = np.arange(len(u_history)) * dt

        plt.figure(figsize=(12, 6))

        # 1. Trajectory (X-Y plane)
        plt.subplot(1, 2, 1)
        plt.plot(ref_traj[:, 0], ref_traj[:, 1], "r--", label="Reference")
        plt.plot(x_history[:, 0], x_history[:, 1], "b-", label="Actual")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Trajectory Tracking")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        # 2. Control Inputs
        plt.subplot(1, 2, 2)
        plt.plot(time_steps, u_history[:, 0], "g-", label="Velocity (v)")
        plt.plot(time_steps, u_history[:, 1], "k-", label="Angular Vel (w)")
        plt.xlabel("Time [s]")
        plt.ylabel("Control Input")
        plt.title("Control Efforts")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
