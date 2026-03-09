import numpy as np
import cupy as cp
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _numba_multiplier(A, B):
    n, m = A.shape
    p = B.shape[1]
    C = np.zeros((n, p))
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


class MatrixMultiplier:

    def __init__(self, A: np.array, B: np.array):
        self.A = A
        self.B = B

        if self.A.shape[1] != self.B.shape[0]:
            raise ValueError(
                f"Incompatible shapes: {self.A.shape} and {self.B.shape}"
            )

    def multiply_naive(self):
        n, m = self.A.shape
        p = self.B.shape[1]
        C = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C[i, j] += self.A[i, k] * self.B[k, j]
        return C

    def multiply_numpy(self):
        return np.dot(self.A, self.B)

    def multiply_numba(self):
        return _numba_multiplier(self.A, self.B)

    def multiply_gpu(self):
        with cp.cuda.Device(0):
            A_gpu = cp.asarray(self.A)
            B_gpu = cp.asarray(self.B)

            C_gpu = cp.dot(A_gpu, B_gpu)

            cp.cuda.Stream.null.synchronize()

            return cp.asnumpy(C_gpu)
