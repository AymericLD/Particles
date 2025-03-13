from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class StreamFunction(ABC):
    L: float

    @abstractmethod
    def stream_function(self, *args: float) -> float: ...

    @abstractmethod
    def derivative(self): ...

    @abstractmethod
    def plot_stream_function_contours(
        self,
        x_domain: NDArray,
        y_domain: NDArray,
        x: NDArray,
        y: NDArray,
        savefig: bool,
        filepath: str,
    ): ...


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
        return self.psi_0 * (
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
        self,
        x_domain: NDArray,
        y_domain: NDArray,
        x: NDArray,
        y: NDArray,
        savefig: bool,
        filepath: str,
    ):
        X, Y = np.meshgrid(x_domain, y_domain)
        Z = np.vectorize(self.stream_function)(X, Y)

        plt.figure(figsize=(8, 6))
        # For plotting the contour of the stream function at levels given by the trajectories
        # W = np.vectorize(self.stream_function)(x, y)
        # levels = np.sort(W)
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(contour, label="Stream function value (ψ)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines in moving frame")
        if savefig:
            plt.savefig(f"{filepath}Streamlines.png")
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
        psi = psi_0 * (1 - sp.tanh((y - y_c) / (width / sp.cos(alpha)))) + c_x * y
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
        self,
        x_domain: NDArray,
        y_domain: NDArray,
        x: NDArray,
        y: NDArray,
        savefig: bool,
        filepath: str,
    ):
        X, Y = np.meshgrid(x_domain, y_domain)
        Z = np.vectorize(self.stream_function)(X, Y)

        plt.figure(figsize=(8, 6))
        # For plotting the contour of the stream function at levels given by the trajectories
        # W = np.vectorize(self.stream_function)(x, y)
        # levels = np.sort(W)
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(contour, label="Stream function value (ψ)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines in moving frame")
        if savefig:
            plt.savefig(f"{filepath}Streamlines.png")
        plt.show()


@dataclass(frozen=True)
class Cellular_Flow(StreamFunction):
    epsilon: float
    L: float
    alpha: float
    phi: float

    @cached_property
    def k(self):
        return 2 * np.pi / self.L

    @cached_property
    def omega(self):
        return np.pi * self.alpha / self.L

    def stream_function(self, x, y, t) -> float:
        a = np.sin((self.k * (x - self.epsilon * np.sin(self.omega * t))))
        b = np.sin(self.k * (y - self.epsilon * np.sin(self.omega * t + self.phi)))
        return a * b * self.alpha / self.k

    @cached_property
    def derivative(self):
        x, y, t = sp.symbols("x y t")
        omega, epsilon, alpha, phi, k = sp.symbols("omega epsilon alpha phi k")
        a = sp.sin((self.k * (x - self.epsilon * sp.sin(self.omega * t))))
        b = sp.sin(self.k * (y - self.epsilon * sp.sin(self.omega * t + self.phi)))
        psi = a * b * alpha / k
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        u_fixed = u.subs(
            {
                omega: self.omega,
                epsilon: self.epsilon,
                alpha: self.alpha,
                phi: self.phi,
                k: self.k,
            }
        )
        v_fixed = v.subs(
            {
                omega: self.omega,
                epsilon: self.epsilon,
                alpha: self.alpha,
                phi: self.phi,
                k: self.k,
            }
        )
        v_func = sp.lambdify((x, y, t), v_fixed, modules=["numpy"])
        u_func = sp.lambdify((x, y, t), u_fixed, modules=["numpy"])
        return (u_func, v_func)

    def plot_stream_function_contours(
        self,
        x_domain: NDArray,
        y_domain: NDArray,
        x: NDArray,
        y: NDArray,
        savefig: bool,
        filepath: str,
    ):
        return


@dataclass(frozen=True)
class Bower_StreamFunction_with_inertial_waves(StreamFunction):
    """Streamfunction given by the Bower Article with additionnal inertial waves"""

    psi_0: float
    A: float
    L: float
    width: float
    c_x: float
    f: float
    gamma: float

    @cached_property
    def k(self):
        return 2 * np.pi / self.L

    def y_c(self, x, t):
        return self.A * np.sin(self.k * (x - self.c_x * t))

    def alpha(self, x, t):
        return np.arctan(self.A * self.k * np.cos(self.k * (x - self.c_x * t)))

    def stream_function(self, x, y, t):
        return self.psi_0 * (
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
        psi_0, A, k, width, c_x, f, gamma = sp.symbols("psi_0 A k width c_x f gamma")
        y_c = A * sp.sin(k * (x - c_x * t))
        alpha = sp.atan(A * k * sp.cos(k * (x - c_x * t)))
        psi = psi_0 * (1 - sp.tanh((y - y_c) / (width / sp.cos(alpha))))
        dpsi_dx = sp.diff(psi, x)
        dpsi_dy = sp.diff(psi, y)
        v = dpsi_dx * (1 - gamma * sp.cos(f * t)) - dpsi_dy * gamma * sp.sin(f * t)
        u = -dpsi_dy * (1 - gamma * sp.cos(f * t)) - dpsi_dx * gamma * sp.sin(f * t)
        v_fixed = v.subs(
            {
                psi_0: self.psi_0,
                A: self.A,
                k: self.k,
                width: self.width,
                c_x: self.c_x,
                f: self.f,
                gamma: self.gamma,
            }
        )
        u_fixed = u.subs(
            {
                psi_0: self.psi_0,
                A: self.A,
                k: self.k,
                width: self.width,
                c_x: self.c_x,
                f: self.f,
                gamma: self.gamma,
            }
        )
        v_func = sp.lambdify((x, y, t), v_fixed, modules=["numpy"])
        u_func = sp.lambdify((x, y, t), u_fixed, modules=["numpy"])
        return (u_func, v_func)

    def plot_stream_function_contours(
        self,
        x_domain: NDArray,
        y_domain: NDArray,
        x: NDArray,
        y: NDArray,
        savefig: bool,
        filepath: str,
    ):
        X, Y = np.meshgrid(x_domain, y_domain)
        Z = np.vectorize(self.stream_function)(X, Y)

        plt.figure(figsize=(8, 6))
        # For plotting the contour of the stream function at levels given by the trajectories
        # W = np.vectorize(self.stream_function)(x, y)
        # levels = np.sort(W)
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(contour, label="Stream function value (ψ)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines in moving frame")
        if savefig:
            plt.savefig(f"{filepath}Streamlines.png")
        plt.show()


@dataclass(frozen=True)
class Axisymmetric_Vortex(StreamFunction):
    """Streamfunction of an axisymmetric vortex with given vorticity, with inertial waves"""

    L: float
    R: float
    zeta: float
    gamma: float

    def stream_function(self, r, theta, t):
        if r < self.R:
            return (self.zeta * r**2) / 4
        else:
            return ((self.zeta * self.R**2) / 2) * np.log(r / self.R) + (
                self.zeta * self.R**2
            ) / 4

    @cached_property
    def derivative(self):
        x, y, t = sp.symbols("x y t")
        L, R, zeta, gamma = sp.symbols("L R zeta gamma")
        r = sp.sqrt(x**2 + y**2)
        psi = sp.Piecewise(
            ((zeta * r**2) / 4, r < R),
            (((zeta * R**2) / 2) * sp.ln(r / R) + (zeta * R**2) / 4, r >= R),
        )
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        u += gamma * sp.cos(2 * sp.pi * zeta * t)
        v += gamma * sp.sin(2 * sp.pi * zeta * t)
        v_fixed = v.subs({L: self.L, R: self.R, zeta: self.zeta, gamma: self.gamma})
        u_fixed = u.subs({L: self.L, R: self.R, zeta: self.zeta, gamma: self.gamma})
        v_func = sp.lambdify((x, y, t), v_fixed, modules=["numpy"])
        u_func = sp.lambdify((x, y, t), u_fixed, modules=["numpy"])
        return (u_func, v_func)

    @cached_property
    def derivative_verification(self):
        x, y, t = sp.symbols("x y t")
        L, R, zeta, gamma = sp.symbols("L R zeta gamma")
        r = sp.sqrt(x**2 + y**2)
        psi = sp.Piecewise(
            ((zeta * r**2) / 4, r < R),
            (((zeta * R**2) / 2) * sp.ln(r / R) + (zeta * R**2) / 4, r >= R),
        )
        v = sp.diff(psi, x)
        u = -sp.diff(psi, y)
        v += gamma * sp.sin(2 * sp.pi * zeta * t)
        u += gamma * sp.cos(2 * sp.pi * zeta * t)
        print("u=", u)
        print("v=", v)

    def plot_stream_function_contours(
        self,
        x_domain: NDArray,
        y_domain: NDArray,
        x: NDArray,
        y: NDArray,
        savefig: bool,
        filepath: str,
    ):
        X, Y = np.meshgrid(x_domain, y_domain)
        Z = np.vectorize(self.stream_function)(X, Y)

        plt.figure(figsize=(8, 6))
        # For plotting the contour of the stream function at levels given by the trajectories
        # W = np.vectorize(self.stream_function)(x, y)
        # levels = np.sort(W)
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
        plt.colorbar(contour, label="Stream function value (ψ)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Streamlines in moving frame")
        if savefig:
            plt.savefig(f"{filepath}Streamlines.png")
        plt.show()
