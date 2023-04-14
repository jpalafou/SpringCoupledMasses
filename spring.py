from numpy.random import rand
import numpy as np


class SpringCoupledMasses:
    def __init__(self, m, pairs, stiffness=None, damping=None):
        """
        solver for system of point masses connected by damped springs

        args:
            m           list of masses
            pairs       list of tuples of mass indices
            stiffness   list of stiffness values
            damping     list of damping values
        returns:
            K           stiffness matrix
            B           damping matrix
            A           linear operator
        """
        n = len(m)
        self.n = n
        self.pairs = pairs
        self.m = m
        self._get_stiffness_and_damping(pairs, stiffness, damping)
        # mass matrix
        minv = np.diag(np.reciprocal(np.array(m), dtype="float64"))
        # stiffness and damping operators
        ones = np.ones(n)
        A_stiffness = minv @ (self.K - np.diag(self.K @ ones))
        A_damping = minv @ (self.B - np.diag(self.B @ ones))
        # combined linear operator
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.identity(n)
        A[n:, :n] = A_stiffness
        A[n:, n:] = A_damping
        self.A = A
        # eigenvalues and stiffness
        self.eigvals = np.linalg.eig(A)[0]
        self.numerical_stiffness = np.max(np.abs(self.eigvals)) / np.min(
            np.abs(self.eigvals)
        )

    def _get_stiffness_and_damping(self, pairs, stiffness, damping):
        K = np.zeros((self.n, self.n))  # stiffness matrix
        B = np.zeros((self.n, self.n))  # damping matrix
        if stiffness is not None:
            for i, k in zip(pairs, stiffness):
                K[i] = k
                K[i[::-1]] = k
        if damping is not None:
            for i, b in zip(pairs, damping):
                B[i] = b
                B[i[::-1]] = b
        self.K = K
        self.B = B

    def set_initial_condition(self, x0=None, v0=None, d=2):
        """
        args:
            x0  array of shape (n,d) where d is dimensionality
            v0  like x0
        """
        if x0 is None:
            x0 = 2 * rand(self.n, d) - 1
        if v0 is None:
            v0 = 2 * rand(self.n, d) - 1
        self.x0 = x0  # d-dimensional position
        self.v0 = v0  # d-dimensional velocity

    def rk4(self, T=1.0, timesteps=101, frames=11):
        """
        args:
            T           integration stopping time
            timesteps   number of discretized time values
            frames      number of time values to save at
        """
        # intialize storage
        self.t = np.zeros(frames)
        U0 = np.concatenate((self.x0, self.v0))
        U1 = np.concatenate((np.zeros(self.x0.shape), np.zeros(self.v0.shape)))
        self.X = np.repeat(np.zeros(self.x0.shape)[None, ...], frames, axis=0)
        self.V = np.repeat(np.zeros(self.v0.shape)[None, ...], frames, axis=0)
        self.X[0] = self.x0
        self.V[0] = self.v0
        # indices to save at
        idx_log = np.linspace(0, timesteps - 1, frames).astype(int)
        # time vector used in solution
        _t = np.linspace(0, T, timesteps)
        # initialize temporary k arrays
        k1 = np.zeros(U1.shape)
        k2 = np.zeros(U1.shape)
        k3 = np.zeros(U1.shape)
        k4 = np.zeros(U1.shape)
        n = self.n  # index for grabbing position and velocity
        for i in range(timesteps - 1):  # iterate through timesteps
            dt = _t[i + 1] - _t[i]
            for j in range(U1.shape[-1]):  # dimension
                k1[:, j] = self.A @ U0[:, j]
                k2[:, j] = self.A @ (U0[:, j] + 0.5 * dt * k1[:, j])
                k3[:, j] = self.A @ (U0[:, j] + 0.5 * dt * k2[:, j])
                k4[:, j] = self.A @ (U0[:, j] + dt * k3[:, j])
            U1 = U0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            if i + 1 in idx_log:
                self.t[np.where(idx_log == i + 1)] = _t[i + 1]
                self.X[np.where(idx_log == i + 1)] = U1[:n]
                self.V[np.where(idx_log == i + 1)] = U1[n:]
            # reset values
            U0 = U1
