import matplotlib.pyplot as plt
import numpy as np
import os


def plot_data_generation(
    u_velocity_raw, u_velocity_smooth, u_angular_raw, u_angular_smooth
):
    """
    Plots the raw vs smoothed control inputs used for data generation.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(u_velocity_raw, label="Raw u[0] (velocity)", alpha=0.6, linestyle="--")
    plt.plot(u_velocity_smooth, label="Smoothed u[0] (velocity)", linewidth=2)

    plt.plot(u_angular_raw, label="Raw u[1] (angular)", alpha=0.6, linestyle="--")
    plt.plot(u_angular_smooth, label="Smoothed u[1] (angular)", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Control Input")
    plt.legend()
    plt.title("Random Control Inputs and Smoothed Signals")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


class Visualizer:
    """
    Handles plotting and saving of simulation results.
    """

    @staticmethod
    def plot_simulation_results(ref_traj, x_history, u_history, dt, save_dir="results"):
        """
        Generates and saves standard engineering plots for the MPC simulation.

        Args:
            ref_traj: Reference trajectory array (N, 3)
            x_history: Actual state history array (N, 3)
            u_history: Control input history array (N, 2)
            dt: Time step
            save_dir: Directory to save PDF figures
        """
        os.makedirs(save_dir, exist_ok=True)

        x_history = np.array(x_history)
        u_history = np.array(u_history)
        ref_traj = np.array(ref_traj)

        min_len = min(len(ref_traj), len(x_history))
        time_axis = np.arange(min_len) * dt

        error_vec = x_history[:min_len, :2] - ref_traj[:min_len, :2]
        error_norm = np.linalg.norm(error_vec, axis=1)

        # --- Figure 1: Trajectory Tracking ---
        plt.figure(figsize=(8, 8))
        plt.plot(ref_traj[:, 0], ref_traj[:, 1], "r--", linewidth=2, label="Reference")
        plt.plot(
            x_history[:, 0], x_history[:, 1], "b-", linewidth=2, label="MPC Tracking"
        )
        plt.scatter(
            ref_traj[0, 0], ref_traj[0, 1], color="g", marker="o", s=100, label="Start"
        )

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axis("equal")
        plt.title("Figure 1: Trajectory Tracking Performance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "figure_1_trajectory.pdf"))
        plt.close()
        print(f"[Visualizer] Saved figure_1_trajectory.pdf")

        # --- Figure 2: Control Inputs ---
        plt.figure(figsize=(10, 5))
        u_time_axis = np.arange(len(u_history)) * dt

        plt.step(
            u_time_axis,
            u_history[:, 0],
            where="post",
            label="Velocity (v)",
            linewidth=2,
        )
        plt.step(
            u_time_axis,
            u_history[:, 1],
            where="post",
            label="Angular Vel (w)",
            linewidth=2,
        )

        plt.xlabel("Time [s]")
        plt.ylabel("Control Input")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title("Figure 2: Control Inputs")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "figure_2_controls.pdf"))
        plt.close()
        print(f"[Visualizer] Saved figure_2_controls.pdf")

        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, error_norm, "k-", linewidth=2, label="Error Norm ($L_2$)")

        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title("Figure 3: Tracking Error over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "figure_3_error.pdf"))
        plt.close()
        print(f"[Visualizer] Saved figure_3_error.pdf")
