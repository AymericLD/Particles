import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from dataclasses import dataclass
from functools import cached_property
from numpy.typing import NDArray
from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
    Cellular_Flow,
    Bower_StreamFunction_with_inertial_waves,
)
from Particles import Evolution_Particles
from Dispersion_Stats import One_Particle_Stats, Two_Particles_Stats
from Spectral_Analysis import Spectral_Particles_Analysis


@dataclass(frozen=True)
class Plots:
    """Designed to make plots of various quantities for the Bower flow"""

    time_step: float
    t_final: float
    num_pairs: int
    f: float
    Gamma: NDArray
    initial_separation: float
    x_min_init: float
    x_max_init: float
    y_min_init: float
    y_max_init: float

    def relative_dispersion_gamma(self, savefig: bool, filepath: str):
        plt.figure(figsize=(10, 6))
        for g in self.Gamma:
            stream_function = Bower_StreamFunction_with_inertial_waves(
                psi_0=4e3, A=50, L=400, width=40, c_x=10, f=8.64, gamma=g
            )
            Particles = Evolution_Particles(
                stream_function=stream_function,
                time_step=self.time_step,
                t_final=self.t_final,
                num_pairs=self.num_pairs,
                x_min_init=self.x_min_init,
                x_max_init=self.x_max_init,
                y_min_init=self.y_min_init,
                y_max_init=self.y_max_init,
                x_min=0,
                x_max=2 * stream_function.L,
                y_min=-250,
                y_max=250,
                initial_separation=self.initial_separation,
            )
            x, y, times = Particles.solve_ODE(PBC=False)
            Stats = Two_Particles_Stats(
                stream_function=stream_function,
                x=x,
                y=y,
                times=times,
                num_particles=2 * self.num_pairs,
                initial_separation=self.initial_separation,
            )
            # savefig=2 is used for plotting several spectra on the same figure
            Stats.plot_time_evolution_dispersion(
                filepath=filepath, savefig=2, f=self.f, gamma=g
            )
        plt.loglog(times, 1e-6 * (times / 1e-2) ** 2, "k--", label=r"$t^2$")
        plt.xlabel("Time (days)")
        plt.ylabel("Relative Dispersion")
        plt.title("Relative Dispersion as a function of time with varying gamma")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"{filepath}Relative_Dispersion,stream_function={stream_function}f={self.f}gamma={self.Gamma}num_particles={2 * self.num_pairs}.png"
            )
        else:
            plt.show()

    def velocity_spectrum_gamma(self, savefig: bool, filepath: str):
        plt.figure(figsize=(10, 6))
        for g in self.Gamma:
            stream_function = Bower_StreamFunction_with_inertial_waves(
                psi_0=4e3, A=50, L=400, width=40, c_x=10, f=8.64, gamma=g
            )
            Particles = Evolution_Particles(
                stream_function=stream_function,
                time_step=self.time_step,
                t_final=self.t_final,
                num_pairs=self.num_pairs,
                x_min_init=self.x_min_init,
                x_max_init=self.x_max_init,
                y_min_init=self.y_min_init,
                y_max_init=self.y_max_init,
                x_min=0,
                x_max=2 * 400,
                y_min=-250,
                y_max=250,
                initial_separation=self.initial_separation,
            )
            Spec = Spectral_Particles_Analysis(
                stream_function=stream_function,
                time_step=self.time_step,
                t_final=self.t_final,
                num_particles=2 * self.num_pairs,
                f=self.f,
            )
            u, v, times = Particles.velocity_profiles
            # savefig=2 is used for plotting several spectra on the same figure
            Spec.plot_velocity_spectrum(
                u, v, filepath=filepath, savefig=2, f=self.f, gamma=g
            )
        plt.axvline(
            x=stream_function.k * stream_function.c_x,
            color="r",
            linestyle="--",
            linewidth=2,
            label=r"$kc_x$",
        )
        plt.axvline(
            x=self.f / (2 * np.pi),
            color="r",
            linestyle="--",
            linewidth=2,
            label=r"$f$",
        )
        plt.xlabel("Frequency (d^-1)")
        plt.ylabel("Spectral Density")
        plt.title("Velocity Spectral Density in frequency with varying gamma")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"{filepath}Velocity_Spectrum,f={self.f}gamma={self.Gamma}num_particles={2 * self.num_pairs}.png"
            )
        else:
            plt.show()
