import numpy as np
import time

# --- Engineering Modules ---
from src.system import AnalyticalRobot, NeuralRobot
from src.trajectory import TrajectoryGenerator
from src.control import MPCController, MPCParams
from src.visualization import Visualizer


def main():
    # ==========================
    # 1. Configuration & Setup
    # ==========================
    DT = 0.05
    TOTAL_TIME = 20.0  # Simulation duration
    HORIZON = 25  # MPC Prediction Horizon

    # Define MPC Weights (Tuning Parameters)
    Q = np.diag([10.0, 10.0, 1.0])  # State Cost: [x, y, theta]
    R = np.diag([1, 1])  # Control Cost: [v, w]
    P = np.diag([0.0, 0.0, 0.0])  # Terminal Cost

    # Initialize Parameters
    mpc_params = MPCParams(Q=Q, R=R, P=P, dt=DT, horizon=HORIZON)

    # Initialize Components
    plant = AnalyticalRobot(dt=DT)

    controller_model = NeuralRobot(dt=DT)
    controller = MPCController(system=controller_model, params=mpc_params)

    # ==========================
    # 2. Trajectory Generation
    # ==========================
    print("Generating Reference Trajectory...")
    full_ref_traj = TrajectoryGenerator.generate_circle(
        total_time=TOTAL_TIME + (HORIZON * DT) + 5.0, dt=DT, radius=5.0, omega=0.3
    )

    # ==========================
    # 3. Simulation Loop
    # ==========================
    print(f"Starting Simulation ({TOTAL_TIME}s)...")

    # Initial State (Start at the first reference point)
    x_current = full_ref_traj[0].copy()
    x_pred_model = x_current.copy()

    # Data logging
    x_history = [x_current]
    u_history = []

    start_time = time.time()
    num_steps = int(TOTAL_TIME / DT)

    for k in range(num_steps):
        error_correction = x_current - x_pred_model
        ref_window = full_ref_traj[k : k + HORIZON]

        try:
            u_opt, _ = controller.solve(
                x_current, ref_window, error_correction=error_correction
            )
        except Exception as e:
            print(f"MPC Solver failed at step {k}: {e}")
            u_opt = np.array([0.0, 0.0])  # Safe fallback control

        x_next_real, _ = plant.step(x_current, u_opt)
        x_next_model_guess, _ = controller_model.step(x_current, u_opt)

        u_history.append(u_opt)
        x_history.append(x_next_real)

        x_current = x_next_real
        x_pred_model = x_next_model_guess

        # Logging progress
        if k % 20 == 0:
            print(
                f"Step {k:3d}/{num_steps} | State: {x_current[:2]} | Control: {u_opt}"
            )

    elapsed = time.time() - start_time
    print(f"Simulation Complete. Time elapsed: {elapsed:.2f}s")

    # ==========================
    # 4. Visualization
    # ==========================
    print("Generating Plots...")
    Visualizer.plot_simulation_results(
        ref_traj=full_ref_traj[: len(x_history)],
        x_history=x_history,
        u_history=u_history,
        dt=DT,
        save_dir="results",
    )
    print("Done. Check the 'results' folder.")


if __name__ == "__main__":
    main()
