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

    psi_0: float
    A: float
    L: float
    width: float
    c_x: float
    time_step: float
    t_final: float
    particle_number: int
    f: float
    Gamma: NDArray
    initial_separation: float

    def relative_dispersion_gamma(self, savefig: bool, filepath: str):
        plt.figure(figsize=(10, 6))
        for g in self.Gamma:
            Inertial_waves = Bower_StreamFunction_with_inertial_waves(
                psi_0=self.psi_0,
                A=self.A,
                L=self.L,
                width=self.width,
                c_x=self.c_x,
                f=self.f,
                gamma=g,
            )
            Particles = Evolution_Particles(
                stream_function=Inertial_waves,
                time_step=self.time_step,
                t_final=self.t_final,
                particle_number=self.particle_number,
                x_min=0,
                x_max=2 * Inertial_waves.L,
                y_min=-250,
                y_max=250,
            )
            x, y, times = Particles.solve_ODE(PBC=False)
            Stats = Two_Particles_Stats(
                x=x,
                y=y,
                times=times,
                num_particles=self.particle_number,
                initial_separation=self.initial_separation,
            )
            # savefig=2 is used for plotting several spectra on the same figure
            Stats.plot_time_evolution_dispersion(
                filepath=filepath, savefig=2, f=self.f, gamma=g
            )

        plt.xlabel("Time (days)")
        plt.ylabel("Relative Dispersion")
        plt.title("Relative Dispersion as a function of time with varying gamma")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"{filepath}Relative_Dispersion,f={self.f}gamma={self.Gamma}num_particles={self.particle_number}.png"
            )
        else:
            plt.show()

    def velocity_spectrum_gamma(self, savefig: bool, filepath: str):
        plt.figure(figsize=(10, 6))
        for g in self.Gamma:
            Inertial_waves = Bower_StreamFunction_with_inertial_waves(
                psi_0=self.psi_0,
                A=self.A,
                L=self.L,
                width=self.width,
                c_x=self.c_x,
                f=self.f,
                gamma=g,
            )
            Particles = Evolution_Particles(
                stream_function=Inertial_waves,
                time_step=self.time_step,
                t_final=self.t_final,
                particle_number=self.particle_number,
                x_min=0,
                x_max=2 * Inertial_waves.L,
                y_min=-250,
                y_max=250,
            )
            Spec = Spectral_Particles_Analysis(
                time_step=self.time_step,
                t_final=self.t_final,
                num_particles=self.particle_number,
                f=self.f,
            )
            u, v = Particles.velocity_profiles
            # savefig=2 is used for plotting several spectra on the same figure
            Spec.plot_velocity_spectrum(
                u, v, filepath=filepath, savefig=2, f=self.f, gamma=g
            )
        plt.xlabel("Frequency (d^-1)")
        plt.ylabel("Spectral Density")
        plt.title("Velocity Spectral Density in frequency with varying gamma")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"{filepath}Velocity_Spectrum,f={self.f}gamma={self.Gamma}num_particles={self.particle_number}.png"
            )
        else:
            plt.show()
