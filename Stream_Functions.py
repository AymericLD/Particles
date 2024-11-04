from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class StreamFunction(ABC):
    @abstractmethod
    def stream_function(self, *args: float) -> float: ...


@dataclass(frozen=True)
class Bower_StreamFunction(StreamFunction):
    """Streamfunction given by the Bower Article"""

    psi_0: float
    A: float
    L: float
    width: float
    c_x: float

    @cached_property
    def k(self):
        return 2 * np.pi / self.L

    def y_c(self, x, t):
        return self.A * np.sin(self.k * (x - self.c_x * t))

    def alpha(self, x, t):
        return np.arctan(self.A * self.k * np.cos(self.k * (x - self.c_x * t)))

    def stream_function(self, x, y, t):
        return self.psi_0(
            1 - np.tanh((y - self.y_c(x, t)) / (self.width / np.cos(self.alpha(x, t))))
        )

    @cached_property
    def derivative_verification(self):
        x, y, t = sp.symbols("x y t")
        psi_0, A, k, width, c_x = sp.symbols("psi_0 A k width c_x")
        y_c = A * sp.sin(k * (x - c_x * t))
        alpha = sp.atan(A * k * sp.cos(k * (x - c_x * t)))
        psi = psi_0 * (1 - sp.tanh((y - y_c) / (width / sp.cos(alpha))))
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        print("u=", u)
        print("v=", v)

    @cached_property
    def derivative(self):
        x, y, t = sp.symbols("x y t")
        psi_0, A, k, width, c_x = sp.symbols("psi_0 A k width c_x")
        y_c = A * sp.sin(k * (x - c_x * t))
        alpha = sp.atan(A * k * sp.cos(k * (x - c_x * t)))
        psi = psi_0 * (1 - sp.tanh((y - y_c) / (width / sp.cos(alpha))))
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        v_fixed = v.subs(
            {psi_0: self.psi_0, A: self.A, k: self.k, width: self.width, c_x: self.c_x}
        )
        u_fixed = u.subs(
            {psi_0: self.psi_0, A: self.A, k: self.k, width: self.width, c_x: self.c_x}
        )
        v_func = sp.lambdify((x, y, t), v_fixed, modules=["numpy"])
        u_func = sp.lambdify((x, y, t), u_fixed, modules=["numpy"])
        return (u_func, v_func)

    def plot_stream_function_contours(
        stream_func: StreamFunction, x: NDArray, y: NDArray
    ):
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(stream_func.stream_function)(X, Y)

        plt.figure(figsize=(8, 6))
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(contour, label="Stream function value (ψ)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines in moving frame")
        plt.show()


@dataclass(frozen=True)
class Bower_StreamFunction_in_moving_frame(StreamFunction):
    """Streamfunction given by the Bower Article in the moving frame of the jet"""

    psi_0: float
    A: float
    L: float
    width: float
    c_x: float

    @cached_property
    def k(self):
        return 2 * np.pi / self.L

    def y_c(self, x):
        return self.A * np.sin(self.k * x)

    def alpha(self, x):
        return np.arctan(self.A * self.k * np.cos(self.k * x))

    def stream_function(self, x, y):
        return (
            self.psi_0
            * (1 - np.tanh((y - self.y_c(x)) / (self.width / np.cos(self.alpha(x)))))
            + self.c_x * y
        )

    @cached_property
    def derivative_verification(self):
        x, y = sp.symbols("x y")
        psi_0, A, k, width, c_x = sp.symbols("psi_0 A k width c_x")
        y_c = A * sp.sin(k * x)
        alpha = sp.atan(A * k * sp.cos(k * x))
        psi = psi_0 * (1 - sp.tanh((y - y_c) / (width / sp.cos(alpha))))
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        print("u=", u)
        print("v=", v)

    @cached_property
    def derivative(self):
        x, y, t = sp.symbols("x y t")
        psi_0, A, k, width, c_x = sp.symbols("psi_0 A k width c_x")
        y_c = A * sp.sin(k * x)
        alpha = sp.atan(A * k * sp.cos(k * x))
        psi = psi_0 * (1 - sp.tanh((y - y_c) / (width / sp.cos(alpha))))
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        v_fixed = v.subs(
            {psi_0: self.psi_0, A: self.A, k: self.k, width: self.width, c_x: self.c_x}
        )
        u_fixed = u.subs(
            {psi_0: self.psi_0, A: self.A, k: self.k, width: self.width, c_x: self.c_x}
        )
        v_func = sp.lambdify((x, y, t), v_fixed, modules=["numpy"])
        u_func = sp.lambdify((x, y, t), u_fixed, modules=["numpy"])
        return (u_func, v_func)

    def plot_stream_function_contours(
        stream_func: StreamFunction, x: NDArray, y: NDArray
    ):
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(stream_func.stream_function)(X, Y)

        plt.figure(figsize=(8, 6))
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(contour, label="Stream function value (ψ)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines in moving frame")
        plt.show()