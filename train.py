import os
import sqlite3
import joblib
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import MeanSquaredError

# Import system definitions to know dimensions
from src.system import System3D, System3DLinear, System2D


def get_system(sys_type, dt=0.01):
    if sys_type == "3d_kinematic":
        return System3D(dt)
    elif sys_type == "3d_linear":
        return System3DLinear(dt)
    elif sys_type == "2d":
        return System2D(dt)
    else:
        raise ValueError(f"Unknown system: {sys_type}")


def load_data_from_db(db_path, system_type):
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database {db_path} not found. Run generate_data.py first."
        )

    # 1. Determine columns based on system
    system = get_system(system_type)
    cols_x = [f"x{i}" for i in range(system.n_states)]
    cols_u = [f"u{i}" for i in range(system.n_controls)]
    cols_dx = [f"dx{i}" for i in range(system.n_states)]

    all_cols = cols_x + cols_u + cols_dx
    query_cols = ", ".join(all_cols)

    # 2. Query DB
    print(f"Loading data for {system_type} from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT {query_cols} FROM dynamics_data")
        data = np.array(cursor.fetchall())
    except sqlite3.OperationalError as e:
        print(f"\nSQL Error: {e}")
        print(f"Expected columns: {query_cols}")
        print("Tip: Ensure generate_data.py was run with the same --system argument.\n")
        raise
    finally:
        conn.close()

    # 3. Split into Input (State+Control) and Output (Next State/Derivative)
    nx = system.n_states
    nu = system.n_controls

    X_raw = data[:, :nx]  # States (x0, x1...)
    U_raw = data[:, nx : nx + nu]  # Controls (u0, u1...)
    Y_raw = data[:, nx + nu :]  # Targets (dx0, dx1...)

    # 4. Feature Engineering (Crucial for Kinematic systems)
    if system_type == "3d_kinematic":
        # Transform theta (x2) into cos(theta) and sin(theta)
        # x0, x1, x2 -> x0, x1, cos(x2), sin(x2)
        theta = X_raw[:, 2]
        X_features = np.column_stack(
            (X_raw[:, 0], X_raw[:, 1], np.cos(theta), np.sin(theta))
        )
        # Concatenate with controls
        Inputs = np.hstack((X_features, U_raw))
    else:
        # Linear or 2D systems usually don't need trig wrapping
        Inputs = np.hstack((X_raw, U_raw))

    return Inputs, Y_raw


def build_model(input_dim, output_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(output_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        type=str,
        default="3d_kinematic",
        help="System type: 3d_linear, 3d_kinematic, 2d",
    )
    parser.add_argument("--db", type=str, default="data/generated_data.db")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # Load
        X, y = load_data_from_db(args.db, args.system)

        # Normalize
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        X_norm = input_scaler.fit_transform(X)
        y_norm = output_scaler.fit_transform(y)

        # Save Scalers (Important for inference later)
        joblib.dump(
            input_scaler, os.path.join(MODEL_DIR, f"{args.system}_input_scaler.pkl")
        )
        joblib.dump(
            output_scaler, os.path.join(MODEL_DIR, f"{args.system}_output_scaler.pkl")
        )

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_norm, test_size=0.2
        )

        # Train
        print(f"Training model with Input Dim: {X.shape[1]}, Output Dim: {y.shape[1]}")
        model = build_model(X.shape[1], y.shape[1])
        model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=1,
        )

        # Save
        model_path = os.path.join(MODEL_DIR, f"{args.system}_model.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    except Exception as e:
        print(f"Error: {e}")
