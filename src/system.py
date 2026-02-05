import casadi as ca
import numpy as np


class BaseSystem:
    """
    Base class that handles both Symbolic (for MPC) and Numeric (for Data Gen) dynamics.
    """

    def __init__(self, dt):
        self.dt = dt
        self.nx = self.n_states
        self.nu = self.n_controls

        # 1. Define Symbolic Primitives (One time setup)
        self.sym_x = ca.MX.sym("x", self.nx)
        self.sym_u = ca.MX.sym("u", self.nu)

        # 2. Get Symbolic Expression from Child Class
        self.sym_rhs = self.define_dynamics(self.sym_x, self.sym_u)

        # 3. Create Numerical Function (Auto-compiled)
        # This allows us to run the EXACT same math in Python loops
        self.f_numeric = ca.Function(
            "f_dynamics", [self.sym_x, self.sym_u], [self.sym_rhs]
        )

    @property
    def n_states(self):
        raise NotImplementedError

    @property
    def n_controls(self):
        raise NotImplementedError

    def define_dynamics(self, x, u):
        """
        Must return a CasADi symbolic expression for x_dot.
        Used by the MPC solver.
        """
        raise NotImplementedError

    def compute_dx_numeric(self, x_val, u_val):
        """
        Calculates the numerical derivative for simulation/data generation.
        Input: Numpy arrays
        Output: Numpy array (flattened)
        """
        # Convert inputs to CasADi compatible types (if needed) but Function handles numpy
        res = self.f_numeric(x_val, u_val)
        return np.array(res).flatten()


# --- Concrete Implementations ---


class System2D(BaseSystem):
    @property
    def n_states(self):
        return 2

    @property
    def n_controls(self):
        return 1

    def define_dynamics(self, x, u):
        dx1 = x[1]
        dx2 = -x[0] + ca.sin(u[0])
        return ca.vertcat(dx1, dx2)


class System3D(BaseSystem):
    """Standard Kinematic Car"""

    @property
    def n_states(self):
        return 3

    @property
    def n_controls(self):
        return 2

    def define_dynamics(self, x, u):
        theta = x[2]
        v = u[0]
        w = u[1]

        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = w
        return ca.vertcat(dx, dy, dtheta)


class System3DLinear(BaseSystem):
    """Linear-like structure from your train.ipynb"""

    @property
    def n_states(self):
        return 3

    @property
    def n_controls(self):
        return 2

    def define_dynamics(self, x, u):
        dx = x[1]
        dy = x[2] + u[0]
        dz = -x[0] - 2 * x[1] - 3 * x[2] + u[1]
        return ca.vertcat(dx, dy, dz)
