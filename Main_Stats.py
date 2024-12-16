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
)
from Particles import Evolution_Particles


def main() -> None:
    start_time = time.time()

    # Parameters

    num_particles = 2000
    time_step = 1 / 8
    t_final = 400
    initial_separation = 5

    # Model

    Bower_stream_function = Bower_StreamFunction(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )

    Inertial_Waves = Bower_StreamFunction_with_inertial_waves(
        psi_0=4e3, A=50, L=400, width=40, c_x=10, f=8.64, gamma=1
    )

    # Trajectories

    Particles = Evolution_Particles(
        stream_function=Inertial_Waves,
        time_step=time_step,
        t_final=t_final,
        particle_number=num_particles,
        x_min=0,
        x_max=2 * Bower_stream_function.L,
        y_min=-250,
        y_max=250,
    )

    x, y, times = Particles.solve_ODE(PBC=False)
    Stats = Two_Particles_Stats(x, y, times, num_particles, initial_separation)

    # Plots

    Stats.plot_time_evolution_dispersion

    # plt.ylim(1e-3, 1e4)
    # plt.loglog(np.sqrt(dispersion), K_rel)
    plt.show()

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
