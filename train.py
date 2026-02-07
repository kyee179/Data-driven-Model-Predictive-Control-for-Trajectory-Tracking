import os
import sqlite3
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DB_PATH = os.path.join("data", "system_dynamics_3_2.db")
MODELS_DIR = "models"
MODEL_NAME = "mlp_model"
EPOCHS = 50
BATCH_SIZE = 16


def load_dataset(db_path):
    """Loads and features-engineers data from SQLite."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run generate_data.py first."
        )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Fetch raw: x, y, theta, u_v, u_w | dx, dy, dtheta
    cursor.execute(
        "SELECT x, y, theta, u_velocity, u_angular, dx, dy, dtheta FROM dynamics_data"
    )
    data = np.array(cursor.fetchall())
    conn.close()

    # Inputs: [x, y, theta, v, w] -> Transform theta -> [cos, sin]
    x_pos = data[:, 0:2]  # x, y
    theta = data[:, 2]
    u_ctrl = data[:, 3:5]  # v, w

    cos_theta = np.cos(theta)[:, None]
    sin_theta = np.sin(theta)[:, None]

    X = np.hstack((x_pos, cos_theta, sin_theta, u_ctrl))
    y = data[:, 5:]

    return X, y


def build_mlp(input_dim, output_dim):
    """Creates the Keras Sequential model."""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(output_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def save_artifacts_for_casadi(model, input_scaler, output_scaler, save_dir):
    """
    Saves model weights and scaler params as simple numpy arrays/dicts.
    This allows CasADi to reconstruct the network math symbolically.
    """
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, "mlp_model.h5"))
    print(f"Saved Keras model to {save_dir}")

    weights = [layer.get_weights()[0] for layer in model.layers]
    biases = [layer.get_weights()[1] for layer in model.layers]

    joblib.dump(weights, os.path.join(save_dir, "mlp_weights.pkl"))
    joblib.dump(biases, os.path.join(save_dir, "mlp_biases.pkl"))

    in_params = {"mean": input_scaler.mean_, "std": input_scaler.scale_}
    out_params = {"mean": output_scaler.mean_, "std": output_scaler.scale_}

    joblib.dump(in_params, os.path.join(save_dir, "input_scaler_params.pkl"))
    joblib.dump(out_params, os.path.join(save_dir, "output_scaler_params.pkl"))
    print("Saved weights and scalers for CasADi integration.")


def main():
    print("Loading data...")
    X, y = load_dataset(DB_PATH)

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_norm = input_scaler.fit_transform(X)
    y_norm = output_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=0.2, random_state=42
    )

    print("Training Model...")
    model = build_mlp(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    mse = model.evaluate(X_test, y_test, verbose=0)[0]
    print(f"Test MSE: {mse:.6f}")

    save_artifacts_for_casadi(model, input_scaler, output_scaler, MODELS_DIR)


if __name__ == "__main__":
    main()
