import numpy as np
import argparse
from src.control import MPCController
from src.trajectory import TrajectoryGenerator
from src.visualization import Visualizer
from src.system import System3D, System3DLinear, System2D

# --- CONFIGURATION INTERFACE ---
CONFIG = {
    "dt": 0.05,
    "horizon": 20,
    "total_time": 50.0,
    "system_type": "3d_kinematic",  # Change to '3d_kinematic' or '2d' as needed
    "traj_type": "sin_cos",  # Change to 'circle' as needed
}


def get_system(sys_type, dt):
    if sys_type == "3d_kinematic":
        return System3D(dt)
    elif sys_type == "3d_linear":
        return System3DLinear(dt)
    elif sys_type == "2d":
        return System2D(dt)
    else:
        raise ValueError(f"Unknown system type: {sys_type}")


def get_trajectory(traj_type, total_time, dt):
    gen = TrajectoryGenerator()
    if traj_type == "circle":
        return gen.generate_circle_trajectory(total_time, dt)
    elif traj_type == "sin_cos":
        return gen.generate_sin_cos_trajectory(total_time, dt)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


def main():
    # 1. Setup
    dt = CONFIG["dt"]
    system = get_system(CONFIG["system_type"], dt)

    # Weights (Adjust dimensions based on system)
    nx = system.n_states
    nu = system.n_controls
    Q = np.diag([10] * nx)
    R = np.diag([0.1] * nu)  # Lowered R slightly for better tracking

    # 2. Generate Reference
    ref_traj = get_trajectory(CONFIG["traj_type"], CONFIG["total_time"], dt)

    # 3. Initialize Controller
    mpc = MPCController(system, Q, R, CONFIG["horizon"])

    # 4. Simulation Loop
    x_current = np.zeros(nx)
    x_history = [x_current]
    u_history = []

    steps = len(ref_traj) - CONFIG["horizon"]
    print(f"Running Simulation: {CONFIG['system_type']} | {CONFIG['traj_type']}")

    for k in range(steps):
        ref_window = ref_traj[k : k + CONFIG["horizon"]]

        # A. Solve MPC (Uses Symbolic Dynamics internally)
        u_opt = mpc.solve(x_current, ref_window)

        # B. Simulation Step (Uses Numeric Dynamics)
        # FIX: Replaced .dynamics() with .compute_dx_numeric()
        dx_val = system.compute_dx_numeric(x_current, u_opt)

        # Euler Integration
        x_next = x_current + dt * dx_val
        x_current = x_next

        # Log
        x_history.append(x_current)
        u_history.append(u_opt)

        if k % 50 == 0:
            print(f"Step {k}/{steps}")

    # 5. Visualize
    Visualizer.plot_tracking(ref_traj, x_history, u_history, dt)


if __name__ == "__main__":
    main()
