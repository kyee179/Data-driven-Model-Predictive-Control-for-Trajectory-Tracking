import os
import sqlite3
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.system import AnalyticalRobot
from src.visualization import plot_data_generation

# --- Configuration ---
DB_DIR = "data"
DB_NAME = "system_dynamics_3_2.db"
NUM_STEPS = 10000
DT = 0.1
GAUSSIAN_SIGMA = 2
RANDOM_SEED = 42


def setup_database(db_path):
    """Initializes the SQLite database and table."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS dynamics_data (
        step INTEGER PRIMARY KEY,
        x REAL,
        y REAL,
        theta REAL,
        u_velocity REAL,
        u_angular REAL,
        dx REAL,
        dy REAL,
        dtheta REAL
    )
    """
    )
    conn.commit()
    return conn


def generate_controls(num_steps):
    """Generates random smoothed control inputs."""
    np.random.seed(RANDOM_SEED)

    u_velocity_raw = np.random.uniform(-2.5, 2.5, num_steps)
    u_angular_raw = np.random.uniform(-2.0, 2.0, num_steps)

    u_velocity = gaussian_filter1d(u_velocity_raw, GAUSSIAN_SIGMA)
    u_angular = gaussian_filter1d(u_angular_raw, GAUSSIAN_SIGMA)

    return u_velocity, u_angular, u_velocity_raw, u_angular_raw


def main():
    db_path = os.path.join(DB_DIR, DB_NAME)
    print(f"Initializing database at: {db_path}")
    conn = setup_database(db_path)
    cursor = conn.cursor()

    robot = AnalyticalRobot(dt=DT)
    x = np.array([0.0, 0.0, 0.0])

    u_v, u_w, u_v_raw, u_w_raw = generate_controls(NUM_STEPS)

    print(f"Starting simulation for {NUM_STEPS} steps...")

    for k in range(NUM_STEPS):
        u = np.array([u_v[k], u_w[k]])

        x_next, dx = robot.step(x, u)

        cursor.execute(
            """
        INSERT OR REPLACE INTO dynamics_data (step, x, y, theta, u_velocity, u_angular, dx, dy, dtheta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (k, x[0], x[1], x[2], u_v[k], u_w[k], dx[0], dx[1], dx[2]),
        )

        x = x_next

        if k % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()
