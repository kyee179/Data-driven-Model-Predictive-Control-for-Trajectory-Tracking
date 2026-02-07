import casadi as ca
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from src.system import RobotSystem, AnalyticalRobot, NeuralRobot


@dataclass
class MPCParams:
    """Configuration parameters for the MPC Controller."""

    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    dt: float
    horizon: int
    u_min: np.ndarray = np.array([0, -1.5])
    u_max: np.ndarray = np.array([2, 1.5])


class MPCController:
    def __init__(self, system: NeuralRobot, params: MPCParams):
        self.system = system
        self.params = params

    def solve(
        self,
        x0: np.ndarray,
        ref_traj: np.ndarray,
        error_correction: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        opti = ca.Opti()

        p_h = self.params.horizon
        h = self.params.dt
        Q, R, P = self.params.Q, self.params.R, self.params.P

        if error_correction is None:
            error_val = ca.DM([0, 0, 0])
        else:
            error_val = ca.vertcat(*error_correction)

        X = opti.variable(3, p_h)
        U = opti.variable(2, p_h - 1)

        opti.subject_to(X[:, 0] == x0)

        dynamics_func = self.system.get_dynamics_function()

        for k in range(p_h - 1):
            x_k = X[:, k]
            u_k = U[:, k]

            # Prediction
            x_next_model = dynamics_func(x_k, u_k)

            # Constraint: X[k+1] = Model(X[k], U[k]) + ErrorCorrection
            opti.subject_to(X[:, k + 1] == x_next_model + h * error_val)

            # Input Constraints
            opti.subject_to(self.params.u_min <= u_k)
            opti.subject_to(u_k <= self.params.u_max)

            # Safety Constraint
            opti.subject_to(2 * X[:, k + 1].T @ Q @ error_val <= u_k.T @ R @ u_k)

        # Cost Function
        J = 0
        for k in range(p_h):
            ref_idx = min(k, ref_traj.shape[0] - 1)
            ref = ref_traj[ref_idx, :]
            error_state = X[:, k] - ref

            if k < p_h - 1:
                J += ca.mtimes([error_state.T, Q, error_state])
                J += ca.mtimes([U[:, k].T, R, U[:, k]])
            else:
                J += ca.mtimes([error_state.T, P, error_state])

        opti.minimize(J)

        # --- SOLVER OPTIONS (FIXED) ---
        opts = {
            "ipopt": {
                "print_level": 0,
                "warm_start_init_point": "yes",
                "max_iter": 1000,
                "tol": 1,
                "hessian_approximation": "limited-memory",
            },
            "print_time": 0,
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            u_opt = sol.value(U[:, 0])
            x_pred = sol.value(X)
            return u_opt, x_pred
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            return np.array([0.0, 0.0]), np.zeros((3, p_h))
