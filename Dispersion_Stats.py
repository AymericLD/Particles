from dataclasses import dataclass
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
    Cellular_Flow,
)
import time


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

    def absolute_dispersion(self, t: float):
        a = self.average_ensemble(self.position[:, t] - self.position[:, 0]) ** 2
        b = self.average_ensemble((self.position[:, t] - self.position[:, 0]) ** 2)
        return b - a


@dataclass(frozen=True)
class Two_Particles_Stats:
    """Compute two-particles Lagrangian statistical indicators in a given flow"""

    x: NDArray
    y: NDArray
    times: NDArray
    num_particles: int

    @cached_property
    def position(self):
        return np.sqrt(self.x**2 + self.y**2)

    def pairs(self, R_0: float):
        """Return a list containing tuples of particles index which were initially separated with a distance R_0"""
        ps = []
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                if np.abs(self.position[i, 0] - self.position[j, 0]) <= R_0:
                    ps.append([i, j])
        return ps

    def average_ensemble(self, r: NDArray, ps: tuple):
        """It is assumed that r has a shape (Number of Particles, position(time)) and ps is a tuple of pair indexes"""
        separation = []
        for i, j in ps:
            separation.append((r[i] - r[j]) ** 2)
        sep = np.array(separation)
        return np.sum(sep) / self.num_particles

    def relative_dispersion(self, t: float, R_0: float):
        ps = self.pairs(R_0=R_0)
        rel_disp = self.average_ensemble(self.position[:, t], ps)
        return rel_disp
