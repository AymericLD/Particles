from dataclasses import dataclass
from functools import cached_property
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
    Cellular_Flow,
)
from Spectral_Analysis import Spectral_Particles_Analysis
import time

start_time = time.time()


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
        ic[:, 1] = np.linspace(-150, 150, self.particle_number)

        return ic

    @cached_property
    def initial_conditions_pairs(self):
        initial_separation = 25
        if self.particle_number % 2 != 0:
            raise ValueError(
                "Cannot define pairs of particles with an odd number of particles"
            )
        num_pairs = self.particle_number // 2
        ic = np.zeros((self.particle_number, 2))
        ic[:num_pairs, 0] = np.linspace(0, 0, num_pairs)
        ic[:num_pairs, 1] = np.linspace(-150, 150, num_pairs)

        for i in range(num_pairs):
            theta = np.random.uniform(0, 2 * np.pi)
            dx = initial_separation * np.cos(theta)
            dy = initial_separation * np.sin(theta)
            ic[i + num_pairs, 0] = ic[i, 0] + dx
            ic[i + num_pairs, 1] = ic[i, 1] + dy

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
    def solve_ODE(self) -> tuple[NDArray, NDArray, NDArray]:
        ic = (
            self.initial_conditions_pairs.flatten()
        )  # solve_ivp only deals with vectorial arguments
        nb_points = self.t_final / self.time_step
        t_span = [0, self.t_final]
        t_eval = np.linspace(0, self.t_final, round(nb_points))
        sol = solve_ivp(
            fun=self.RHS, t_span=t_span, y0=ic, method="Radau", t_eval=t_eval
        )
        r = sol.y.reshape((self.particle_number, 2, len(t_eval)))
        for i in range(len(t_eval)):
            r[:, :, i] = self.apply_periodic_boundary_conditions(r[:, :, i])
        x = r[:, 0, :]
        y = r[:, 1, :]
        times = sol.t
        return x, y, times

    @property
    def velocity_profiles(self) -> tuple[NDArray, NDArray]:
        x, y, times = self.solve_ODE
        dx_dt = np.diff(x) / np.diff(times)
        dy_dt = np.diff(y) / np.diff(times)
        return (dx_dt, dy_dt)

    @cached_property
    def verification_in_moving_frame(self):
        x, y, times = self.solve_ODE

        if isinstance(self.stream_function, Bower_StreamFunction):
            plt.plot(
                (x - self.stream_function.c_x * times) % self.x_max, y, "."
            )  # Trajectory in moving frame
            # Contour Lines of stream function in moving frame, should be the same as the trajectories
            x_domain = (np.linspace(self.x_min, self.x_max, 100),)
            y_domain = (np.linspace(self.y_min, self.y_max, 100),)
            X, Y = np.meshgrid(x_domain, y_domain)
            Bower_stream_function_in_moving_frame = (
                Bower_StreamFunction_in_moving_frame(
                    psi_0=self.stream_function.psi_0,
                    A=self.stream_function.A,
                    L=self.stream_function.L,
                    width=self.stream_function.width,
                    c_x=self.stream_function.c_x,
                )
            )
            Z = np.vectorize(Bower_stream_function_in_moving_frame.stream_function)(
                X, Y
            )
            levels = np.linspace(Z.min(), Z.max(), 20)
            plt.contour(X, Y, Z, levels=levels)
            plt.show()
            # Values of the stream function in the moving frame along a trajectory
            q = Bower_stream_function_in_moving_frame.stream_function(
                x - self.stream_function.c_x * times, y
            )
            plt.plot(times, q.T)
            plt.show()

        if isinstance(self.stream_function, Bower_StreamFunction_in_moving_frame):
            plt.plot(x % self.x_max, y, ".")  # Trajectory in moving frame
            # Contour Lines of stream function in moving frame, should be the same as the trajectories
            x_domain = (np.linspace(self.x_min, self.x_max, 100),)
            y_domain = (np.linspace(self.y_min, self.y_max, 100),)
            X, Y = np.meshgrid(x_domain, y_domain)
            Z = np.vectorize(self.stream_function.stream_function)(X, Y)
            levels = np.linspace(Z.min(), Z.max(), 20)
            plt.contour(X, Y, Z, levels=levels)
            plt.show()
            # Values of the stream function in the moving frame along a trajectory
            q = self.stream_function.stream_function(x, y)
            plt.plot(times, q.T)
            plt.show()

    def stream_function_along_trajectories(self, x, y, t):
        if isinstance(self.stream_function, Bower_StreamFunction_in_moving_frame):
            psi = self.stream_function.stream_function(x, y)
            return psi
        elif isinstance(self.stream_function, Bower_StreamFunction):
            psi = self.stream_function.stream_function(x, y, t)
            return psi
        else:
            raise ValueError("Not a known stream function")

    def plot_trajectories(
        self, name: str, savefig: bool, streamlines: str, filepath: str
    ):
        x, y, times = self.solve_ODE
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

        if streamlines:
            x_domain = np.linspace(self.x_min, self.x_max, 100)
            y_domain = np.linspace(self.y_min, self.y_max, 100)
            X, Y = np.meshgrid(x_domain, y_domain)
            Z = np.vectorize(self.stream_function.stream_function)(X, Y)

            x_initial = self.initial_conditions[:, 0]
            y_initial = self.initial_conditions[:, 1]
            W = np.vectorize(self.stream_function.stream_function)(x_initial, y_initial)
            levels = np.sort(W)
            contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")
            plt.colorbar(contour, label="Stream function value (ψ)")

        if savefig:
            plt.savefig(
                f"{filepath}Evolution_Lines{name}{self.particle_number},{self.t_final}"
            )
        plt.show()
