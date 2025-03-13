import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import time
from Dispersion_Stats import One_Particle_Stats, Two_Particles_Stats
from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
    Cellular_Flow,
    Bower_StreamFunction_with_inertial_waves,
    Axisymmetric_Vortex,
)
from Particles import Evolution_Particles


def main() -> None:
    start_time = time.time()

    # Backup Folder

    filepath = "Plots/Bower_Flow/Stats/"

    # Parameters

    num_pairs = (50) ** 2
    time_step = 0.01  # 24 / 60
    t_final = 500
    initial_separation = 1e-1
    f = 8.64
    gamma = 1
    L = 400

    # Axisymmetric Vortex

    R = 1
    zeta = 1
    gamma_vortex = 0
    L_vortex = 5

    # Model

    Bower_stream_function = Bower_StreamFunction(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )

    Inertial_Waves = Bower_StreamFunction_with_inertial_waves(
        psi_0=4e3, A=50, L=400, width=40, c_x=10, f=f, gamma=gamma
    )

    axisymmetric_vortex = Axisymmetric_Vortex(
        L=L_vortex, R=R, zeta=zeta, gamma=gamma_vortex
    )

    # Trajectories

    # Particles = Evolution_Particles(
    #     stream_function=Inertial_Waves,
    #     time_step=time_step,
    #     t_final=t_final,
    #     particle_number=num_particles,
    #     x_min=0,
    #     x_max=2 * Inertial_Waves.L,
    #     y_min=-250,
    #     y_max=250,
    #     initial_separation=initial_separation,
    # )

    Particles = Evolution_Particles(
        stream_function=Inertial_Waves,
        time_step=time_step,
        t_final=t_final,
        num_pairs=num_pairs,
        x_min_init=-150,
        x_max_init=150,
        y_min_init=-40,
        y_max_init=40,
        x_min=0,
        x_max=2 * Inertial_Waves.L,
        y_min=-250,
        y_max=250,
        initial_separation=initial_separation,
    )

    x, y, times = Particles.solve_ODE(PBC=False)
    Stats = Two_Particles_Stats(
        Inertial_Waves,
        x,
        y,
        times,
        num_particles=2 * num_pairs,
        initial_separation=initial_separation,
    )

    dispersion = Stats.relative_dispersion

    # Plots

    Stats.plot_time_evolution_dispersion(filepath=filepath, savefig=0, f=f, gamma=gamma)
    Stats.plot_time_evolution_kurtosis(
        filepath=filepath, savefig=True, f=f, gamma=gamma
    )
    Stats.plot_kurtosis_evolution_wrt_R(
        filepath=filepath, savefig=True, f=f, gamma=gamma
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
