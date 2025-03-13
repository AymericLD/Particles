from dataclasses import dataclass
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from Stream_Functions import StreamFunction


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

    stream_function: StreamFunction
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
            (dx**2 + dy**2) ** 2,
            axis=0,
        )
        r_2_avg = (np.mean(dx**2 + dy**2, axis=0)) ** 2

        return r_4 / r_2_avg

    def plot_time_evolution_kurtosis(
        self, filepath: str, savefig: bool, f: float, gamma: float
    ):
        """f is the frequency of the internal waves and gamma is the modulation of the geostrophic current"""
        """ f=0 and gamma=0 for the geostrophic current"""

        K = np.copy(self.kurtosis)
        fig, ax = plt.subplots()
        plt.plot(self.times, K)
        ax.set_xlabel("t (days)")
        ax.set_ylabel(r"$\langle R^4 \rangle$/$\langle R^2 \rangle^2$")
        ax.set_title("Time evolution of kurtosis")
        plt.legend()

        if savefig:
            plt.savefig(
                f"{filepath}Kurtosis,stream_function={self.stream_function},f={f}gamma={gamma}num_particles={self.num_particles}.png"
            )

        else:
            plt.show()

    def plot_kurtosis_evolution_wrt_R(
        self, filepath: str, savefig: bool, f: float, gamma: float
    ):
        """f is the frequency of the internal waves and gamma is the modulation of the geostrophic current"""
        """ f=0 and gamma=0 for the geostrophic current"""

        K = np.copy(self.kurtosis)
        fig, ax = plt.subplots()
        # plt.plot(self.times, K)
        plt.plot(self.relative_dispersion, K)
        # plt.plot(self.times, self.relative_dispersion)
        # plt.plot(self.times, np.sqrt(self.relative_dispersion))
        ax.set_xlabel(r"$\langle R^2 \rangle$")
        ax.set_ylabel(r"$\langle R^4 \rangle$/$\langle R^2 \rangle^2$")
        ax.set_title("Kurtosis Evolution with the Relative Dispersion")
        plt.legend()

        if savefig:
            plt.savefig(
                f"{filepath}Kurtosis_wrt_R,stream_function={self.stream_function}f={f}gamma={gamma}num_particles={self.num_particles}.png"
            )

        else:
            plt.show()

    def plot_time_evolution_dispersion(
        self, filepath: str, savefig: int, f: float, gamma: float
    ):
        dispersion = np.copy(self.relative_dispersion)

        if savefig != 2:
            fig, ax = plt.subplots()

        plt.loglog(
            self.times,
            dispersion - dispersion[0],
            label=f"num_particles={self.num_particles},gamma={gamma}",
        )
        plt.loglog(self.times, 1e-6 * (self.times / 1e-2) ** 2, "k--", label=r"$t^2$")
        plt.loglog(self.times, 1 * (self.times / 10) ** 3, "k--", label=r"$t^3$")
        plt.ylim(1e-15, 1e10)

        if savefig != 2:
            ax.set_xlabel("t (days)")
            ax.set_ylabel(r"Relative Dispersion $(km^2)$")
            ax.set_title("Time evolution of Relative dispersion")
            plt.legend(fontsize=12)

        if savefig == 0:
            plt.savefig(
                f"{filepath}Dispersion_wrt_t,stream_function={self.stream_function}f={f}gamma={gamma}num_particles={self.num_particles}.png"
            )

        elif savefig == 1:
            plt.show()
