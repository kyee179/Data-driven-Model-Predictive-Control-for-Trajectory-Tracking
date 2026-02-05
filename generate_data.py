import numpy as np
import sqlite3
import os
import argparse
from scipy.ndimage import gaussian_filter1d

# Import the systems from your project package
from src.system import System3DLinear, System3D, System2D


def get_system(sys_type, dt):
    """Factory to choose the system for data generation."""
    if sys_type == "3d_linear":
        return System3DLinear(dt)
    elif sys_type == "3d_kinematic":
        return System3D(dt)
    elif sys_type == "2d":
        return System2D(dt)
    else:
        raise ValueError(f"Unknown system: {sys_type}")


def generate_random_controls(steps, n_controls, sigma=2.0):
    """Generate smoothed random control inputs."""
    u_raw = np.random.uniform(-1, 1, size=(steps, n_controls))
    u_smooth = np.zeros_like(u_raw)
    for i in range(n_controls):
        u_smooth[:, i] = gaussian_filter1d(u_raw[:, i], sigma=sigma)
    return u_smooth


def run_data_generation(sys_type, db_path, total_steps=1000, dt=0.05):
    # 1. Initialize the Chosen System
    system = get_system(sys_type, dt)
    print(f"Generating data using system: {sys_type}")

    # 2. Setup State
    x = np.zeros(system.n_states)
    controls = generate_random_controls(total_steps, system.n_controls)

    # 3. Prepare Database
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Dynamic table creation based on state size
    # Creating columns: x0, x1.. u0, u1.. dx0, dx1..
    cols_x = [f"x{i}" for i in range(system.n_states)]
    cols_u = [f"u{i}" for i in range(system.n_controls)]
    cols_dx = [f"dx{i}" for i in range(system.n_states)]

    all_cols = ", ".join(cols_x + cols_u + cols_dx)
    placeholders = ", ".join(["?"] * (len(cols_x) + len(cols_u) + len(cols_dx)))

    create_sql = f"CREATE TABLE dynamics_data (id INTEGER PRIMARY KEY, {all_cols.replace(',', ' REAL,')} REAL)"
    cursor.execute(create_sql)

    # 4. Simulation Loop
    data_buffer = []

    for k in range(total_steps):
        u = controls[k]

        # --- CRITICAL CHANGE ---
        # We assume the system object knows its own physics.
        # We do NOT write equations here.
        dx_val = system.compute_dx_numeric(x, u)
        # -----------------------

        # Record data
        row = np.concatenate([x, u, dx_val])
        data_buffer.append(tuple(row))

        # Euler Integration for next step
        x = x + dx_val * dt

    # 5. Save
    cursor.executemany(
        f"INSERT INTO dynamics_data ({all_cols}) VALUES ({placeholders})", data_buffer
    )
    conn.commit()
    conn.close()
    print(f"Saved {total_steps} rows to {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        type=str,
        default="3d_kinematic",
        help="System type: 3d_linear, 3d_kinematic, 2d",
    )
    parser.add_argument("--out", type=str, default="data/generated_data.db")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_data_generation(args.system, args.out)
