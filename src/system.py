import casadi as ca
import numpy as np
import joblib
import os


class RobotSystem:
    """Base class for robot dynamics."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.x_sym = ca.MX.sym("x", 3)  # [x, y, theta]
        self.u_sym = ca.MX.sym("u", 2)  # [v, omega]

        self.dynamics_func = self.get_dynamics_function()

    def get_dynamics_function(self):
        """Returns the CasADi function f(x, u) -> x_next."""
        raise NotImplementedError

    def normalize_angle(self, theta):
        """Symbolic angle normalization to [-pi, pi]."""
        return (theta + ca.pi) % (2 * ca.pi) - ca.pi

    def step(
        self, x_current: np.ndarray, u_current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the next state.
        Returns: (x_next, dx_computed)
        """
        res = self.dynamics_func(x_current, u_current)
        x_next = np.array(res).flatten()

        dx = (x_next - x_current) / self.dt

        return x_next, dx


class AnalyticalRobot(RobotSystem):
    """
    Physics-based kinematic model.
    """

    def get_dynamics_function(self):
        dx = self.u_sym[0] * ca.cos(self.x_sym[2])
        dy = self.u_sym[0] * ca.sin(self.x_sym[2])
        dtheta = self.u_sym[1]

        x_dot = ca.vertcat(dx, dy, dtheta)

        x_next = self.x_sym + x_dot * self.dt

        return ca.Function("analytical_step", [self.x_sym, self.u_sym], [x_next])


class NeuralRobot(RobotSystem):
    """
    Data-driven model using an MLP approximated within CasADi.
    Requires trained weights and scaler parameters in 'models/' directory.
    """

    def __init__(self, model_dir="models", dt=0.1):
        self.model_dir = model_dir
        self.weights = self._load_artifact("mlp_weights.pkl")
        self.biases = self._load_artifact("mlp_biases.pkl")
        self.input_scaler = self._load_artifact("input_scaler_params.pkl")
        self.output_scaler = self._load_artifact("output_scaler_params.pkl")

        super().__init__(dt)

    def _load_artifact(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}. Run train.py first."
            )
        return joblib.load(path)

    def _feature_engineering(self, x, u):
        """
        Transforms raw state [x, y, theta] and control [v, w]
        into network input features [x, y, cos(theta), sin(theta), v, w].
        """
        theta = x[2]
        features = ca.vertcat(x[0], x[1], ca.cos(theta), ca.sin(theta), u[0], u[1])
        return features

    def get_dynamics_function(self):
        input_features = self._feature_engineering(self.x_sym, self.u_sym)

        in_mean = ca.MX(self.input_scaler["mean"])
        in_std = ca.MX(self.input_scaler["std"])
        norm_input = (input_features - in_mean) / in_std

        layer_out = norm_input

        n_layers = len(self.weights)

        for i in range(n_layers):
            W = ca.MX(self.weights[i])
            b = ca.MX(self.biases[i])

            layer_out = ca.mtimes(layer_out.T, W).T + b

            if i < n_layers - 1:
                layer_out = ca.fmax(0, layer_out)

        out_mean = ca.MX(self.output_scaler["mean"])
        out_std = ca.MX(self.output_scaler["std"])

        delta_norm = layer_out
        delta_state = (delta_norm * out_std) + out_mean

        x_next = self.x_sym + delta_state * self.dt

        return ca.Function("neural_step", [self.x_sym, self.u_sym], [x_next])
