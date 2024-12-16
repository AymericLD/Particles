from dataclasses import dataclass
from functools import cached_property
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class One_Particle_Stats:
    """Compute one-particle Lagrangian statistical indicators in a given flow"""

    x: NDArray
    y: NDArray
    times: NDArray
    num_particles: int

    @cached_property
    def position(self):
        return np.sqrt(self.x**2 + self.y**2)

    def average_ensemble(self, r: NDArray):
        """It is assumed that r has a shape (Number of Particles, position(time))"""
        return np.sum(r, axis=0) / self.num_particles

    @cached_property
    def absolute_dispersion(self):
        a = np.zeros(len(self.times))
        b = np.zeros(len(self.times))
        for t in range(len(self.times)):
            a[t] = self.average_ensemble(self.position[:, t] - self.position[:, 0]) ** 2
            b[t] = self.average_ensemble(
                (self.position[:, t] - self.position[:, 0]) ** 2
            )
        return b - a

    @cached_property
    def absolute_diffusivity(self):
        dA_dt = np.diff(self.absolute_dispersion) / np.diff(self.times)
        return (1 / 2) * dA_dt


@dataclass(frozen=True)
class Two_Particles_Stats:
    """Compute two-particles Lagrangian statistical indicators in a given flow"""

    x: NDArray
    y: NDArray
    times: NDArray
    num_particles: int
    initial_separation: float

    @cached_property
    def position(self):
        return np.sqrt(self.x**2 + self.y**2)

    @cached_property
    def pairs(self):
        """Return a list containing tuples of particles index which were initially separated with a distance R_0"""
        ps = []

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                if (
                    np.sqrt(
                        (self.x[i, 0] - self.x[j, 0]) ** 2
                        + (self.y[i, 0] - self.y[j, 0]) ** 2
                    )
                    <= 1.1 * self.initial_separation
                ):
                    ps.append([i, j])
        return ps

    def average_ensemble(self, x_t: NDArray, y_t: NDArray, ps: tuple):
        """It is assumed that r has a shape (Number of Particles, position(time)) and ps is a tuple of pair indexes"""
        separation = []
        for i, j in ps:
            separation.append((x_t[i] - x_t[j]) ** 2 + (y_t[i] - y_t[j]) ** 2)
        sep = np.array(separation)

        return np.sum(sep) / len(ps)

    @cached_property
    def relative_dispersion(self):
        ps = self.pairs
        rel_disp = np.zeros(len(self.times))
        for t in range(len(self.times)):
            rel_disp[t] = self.average_ensemble(self.x[:, t], self.y[:, t], ps)
        return rel_disp

    @cached_property
    def relative_diffusivity(self):
        dA_dt = np.diff(self.relative_dispersion) / np.diff(self.times)
        return (1 / 2) * dA_dt
