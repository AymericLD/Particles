from dataclasses import dataclass
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


@dataclass(frozen=True)
class One_Particle_Stats:
    """Compute one-particle Lagrangian statistical indicators in a given flow"""

    x: NDArray
    y: NDArray
    times: NDArray
    num_particles: int

    def average_ensemble(self, r: NDArray):
        """It is assumed that r has a shape (Number of Particles, position(time))"""
        return np.sum(r, axis=0) / self.num_particles

    @cached_property
    def absolute_dispersion(self):
        r2_avg = np.mean(self.x - self.x[:, 0], axis=0) ** 2
        avg_r2 = np.mean((self.x - self.x[:, 0]) ** 2, axis=0)
        return avg_r2 - r2_avg

    @cached_property
    def absolute_diffusivity(self):
        dA_dt = np.gradient(self.absolute_dispersion) / np.gradient(self.times)
        return (1 / 2) * dA_dt


@dataclass(frozen=True)
class Two_Particles_Stats:
    """Compute two-particles Lagrangian statistical indicators in a given flow"""

    """The first particle of a pair is assumed to have an index in [0,N_particles/2], the second particle of the pair has and 
    index in [N_particles/2+1,N_particles]"""

    x: NDArray
    y: NDArray
    times: NDArray
    num_particles: int
    initial_separation: float

    @cached_property
    def relative_dispersion(self):
        dispersion = np.mean(
            (
                self.x[: self.num_particles // 2, :]
                - self.x[self.num_particles // 2 :, :]
            )
            ** 2
            + (
                self.y[: self.num_particles // 2, :]
                - self.y[self.num_particles // 2 :, :]
            )
            ** 2,
            axis=0,
        )
        return dispersion

    @cached_property
    def relative_diffusivity(self):
        K_rel = np.gradient(self.relative_dispersion) / np.gradient(self.times)
        return (1 / 2) * K_rel

    @cached_property
    def kurtosis(self):
        dx = self.x[: self.num_particles // 2, :] - self.x[self.num_particles // 2 :, :]
        dy = self.y[: self.num_particles // 2, :] - self.y[self.num_particles // 2 :, :]
        r_4 = np.mean(
            dx**4 + dy**4,
            axis=0,
        )
        r_2_avg = (np.mean(dx**2 + dy**2, axis=0)) ** 2

        return r_4 / r_2_avg

    @cached_property
    def plot_time_evolution_kurtosis(self):
        K = self.kurtosis
        fig, ax = plt.subplots()
        plt.plot(self.times, K)
        ax.set_xlabel("t (days)")
        ax.set_ylabel("kurtosis")
        ax.set_title("Time evolution of kurtosis")

    @cached_property
    def plot_time_evolution_dispersion(self):
        dispersion = self.relative_dispersion
        fig, ax = plt.subplots()
        plt.loglog(self.times, dispersion)
        y = self.times**2
        plt.loglog(self.times, y, "--")
        ax.set_xlabel("t (days)")
        ax.set_ylabel("Relative Dispersion")
        ax.set_title("Time evolution of Relative dispersion")
