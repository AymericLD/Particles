import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from dataclasses import dataclass
from functools import cached_property
from Stream_Functions import StreamFunction


@dataclass(frozen=True)
class Spectral_Particles_Analysis:
    """Spectral Analysis of particles dispersion in a given flow"""

    stream_function: StreamFunction
    time_step: float
    t_final: float
    num_particles: int
    f: float

    def velocity_spectrum(self, u, v):
        N = int(self.t_final / self.time_step) - 1
        window = get_window("blackman", N)

        u_windowed = np.zeros((N, self.num_particles), dtype=complex)
        fft_u = np.zeros((N, self.num_particles), dtype=complex)
        v_windowed = np.zeros((N, self.num_particles), dtype=complex)
        fft_v = np.zeros((N, self.num_particles), dtype=complex)

        for i in range(self.num_particles):
            u_windowed[:, i] = u[i, :] * window
            fft_u[:, i] = np.fft.fft(u_windowed[:, i])
            v_windowed[:, i] = v[i, :] * window
            fft_v[:, i] = np.fft.fft(v_windowed[:, i])

        frequencies_u = np.fft.fftfreq(N, self.time_step)
        freq = frequencies_u[1 : N // 2]
        scale = np.sum(window**2)

        # Spectrum for u

        u_spec_particles = np.abs(fft_u[1 : N // 2]) ** 2
        u_spec = np.sum(u_spec_particles, axis=1) / (self.num_particles * scale)

        # Spectrum for v

        v_spec_particles = np.abs(fft_v[1 : N // 2]) ** 2
        v_spec = np.sum(v_spec_particles, axis=1) / (self.num_particles * scale)

        # Velocity Spectrum

        vel_spec = u_spec + v_spec

        return u_spec, v_spec, vel_spec, freq

    def plot_velocity_spectrum(
        self, u, v, filepath: str, savefig: int, f: float, gamma: float
    ):
        """f is the frequency of the internal waves and gamma is the modulation of the geostrophic current"""
        """ f=0 and gamma=0 for the geostrophic current"""
        """ savefig=0 : figure saved, savefig=1 : figure shown, savefig=2 : no output (used when plotting several spectrum in the same figure)"""

        u_spec, v_spec, vel_spec, freq = self.velocity_spectrum(u, v)

        if savefig != 2:
            plt.figure(figsize=(10, 6))

        # plt.loglog(freq, vel_spec, label="Total")
        # plt.loglog(freq, u_spec, label="Zonal")
        plt.loglog(
            freq, v_spec, label=f"num_particles={self.num_particles},gamma={gamma}"
        )
        # plt.plot([f / (2 * np.pi), f / (2 * np.pi)], [0, 1e12])

        if savefig != 2:
            plt.xlabel(r"Frequency ($d^{-1}$)")
            plt.ylabel("Spectral Density")
            plt.title("Velocity Spectral Density in frequency")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend(fontsize=12)
            plt.tight_layout()

        if savefig == 0:
            plt.savefig(
                f"{filepath}Velocity_Spectrum,stream_function={self.stream_function},f={f}gamma={gamma}num_particles={self.num_particles}.png"
            )
        elif savefig == 1:
            plt.show()
