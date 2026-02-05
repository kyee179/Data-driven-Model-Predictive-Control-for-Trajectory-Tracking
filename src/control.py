import casadi as ca
import numpy as np


class MPCController:
    def __init__(self, system_model, Q, R, horizon):
        self.sys = system_model
        self.Q = Q
        self.R = R
        self.H = horizon

    def solve(self, x0, ref_traj):
        opti = ca.Opti()
        nx = self.sys.n_states
        nu = self.sys.n_controls

        X = opti.variable(nx, self.H + 1)
        U = opti.variable(nu, self.H)

        opti.subject_to(X[:, 0] == x0)

        J = 0
        for k in range(self.H):
            # Cost
            ref = ref_traj[k, :]
            err = X[:, k] - ref
            J += ca.mtimes([err.T, self.Q, err])
            ctrl = U[:, k]
            J += ca.mtimes([ctrl.T, self.R, ctrl])

            # --- CRITICAL: Use the same symbolic definition ---
            # X[:, k] and U[:, k] are symbolic CasADi variables here.
            # define_dynamics returns a symbolic expression.
            dx_sym = self.sys.define_dynamics(X[:, k], U[:, k])

            opti.subject_to(X[:, k + 1] == X[:, k] + self.sys.dt * dx_sym)
            opti.subject_to(opti.bounded(-5.0, U[:, k], 5.0))

        opti.minimize(J)
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            return sol.value(U)[:, 0]
        except RuntimeError:
            return np.zeros(nu)
