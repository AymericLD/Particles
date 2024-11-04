from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import time

start_time = time.time()


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


@dataclass(frozen=True)
class Evolution_Particles:
    """Determine the trajectory of particles in a flow whose streamfunction is given. It is assumed that this streamfunction is periodic in the horizontal direction"""

    stream_function: StreamFunction
    time_step: float
    t_final: float
    particle_number: float
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def __post_init__(self) -> None:
        if not ((self.x_max - self.x_min) % self.stream_function.L == 0):
            raise ValueError(
                "Computationnal domain does not have the flow periodicity in the x direction"
            )

    @cached_property
    def initial_conditions(self):
        ic = np.zeros((self.particle_number, 2))
        ic[:, 0] = np.linspace(0, 0, self.particle_number)
        ic[:, 1] = np.linspace(-50, 50, self.particle_number)

        return ic

    def RHS(self, t: float, y: NDArray) -> NDArray:
        r = y.reshape((self.particle_number, 2))  # Use a N*2 matrix to handle the ODE
        derivatives = np.zeros((self.particle_number, 2))
        derivatives[:, 0] = self.stream_function.derivative[0](r[:, 0], r[:, 1], t)
        derivatives[:, 1] = self.stream_function.derivative[1](r[:, 0], r[:, 1], t)

        return derivatives.flatten()  # solve_ivp only deals with vectorial arguments

    def apply_periodic_boundary_conditions(self, positions: NDArray) -> NDArray:
        positions[:, 0] = self.x_min + (positions[:, 0] - self.x_min) % (
            self.x_max - self.x_min
        )
        positions[:, 1] = self.y_min + (positions[:, 1] - self.y_min) % (
            self.y_max - self.y_min
        )

        return positions

    @property
    def solve_ODE(self) -> tuple[NDArray, NDArray]:
        ic = (
            self.initial_conditions.flatten()
        )  # solve_ivp only deals with vectorial arguments
        nb_points = self.t_final / self.time_step
        t_span = [0, self.t_final]
        t_eval = np.linspace(0, self.t_final, round(nb_points))
        sol = solve_ivp(
            fun=self.RHS, t_span=t_span, y0=ic, method="DOP853", t_eval=t_eval
        )
        r = sol.y.reshape((self.particle_number, 2, len(t_eval)))
        for i in range(len(t_eval)):
            r[:, :, i] = self.apply_periodic_boundary_conditions(r[:, :, i])
        x = r[:, 0, :]
        y = r[:, 1, :]
        return x, y

    def verification_moving_frame(self, x, y):
        if isinstance(self.stream_function, Bower_StreamFunction_in_moving_frame):
            psi = self.stream_function.stream_function(x, y)
            return psi
        else:
            raise ValueError("Not a stream function in the moving frame")

    def plot_trajectories(self, name: str, savefig: bool, filepath: str):
        x, y = self.solve_ODE
        fig, ax = plt.subplots()
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        for i in range(self.particle_number):
            plt.scatter(x[i, :], y[i, :])
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title(
            f"Particle Trajectories (time span={self.t_final} days, time step={self.time_step} days)"
        )
        plt.show()
        if savefig:
            plt.savefig(
                f"{filepath}Evolution_Lines{name}{self.particle_number},{self.t_final}"
            )


def main() -> None:
    Bower_stream_function = Bower_StreamFunction(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )
    Bower_stream_function_in_moving_frame = Bower_StreamFunction_in_moving_frame(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )
    Particles = Evolution_Particles(
        stream_function=Bower_stream_function_in_moving_frame,
        time_step=1 / 8,
        t_final=10,
        particle_number=5,
        x_min=0,
        x_max=2 * Bower_stream_function_in_moving_frame.L,
        y_min=-250,
        y_max=250,
    )

    x_positions, y_positions = Particles.solve_ODE
    print(Particles.verification_moving_frame(x=x_positions, y=y_positions))

    Particles.plot_trajectories(name="Bower", savefig=False, filepath="Plots/")
    Bower_stream_function_in_moving_frame.plot_stream_function_contours(
        x=np.linspace(Particles.x_min, Particles.x_max, 100),
        y=np.linspace(Particles.y_min, Particles.y_max, 100),
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
