from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
    Cellular_Flow,
    Bower_StreamFunction_with_inertial_waves,
)
from Particles import Evolution_Particles
from Spectral_Analysis import Spectral_Particles_Analysis
import time
import numpy as np

start_time = time.time()


def main() -> None:
    # Backup Folder

    filepath = "Plots/Bower_Flow/Spectrum/"

    # Parameters

    time_step = 1 / 8
    t_final = 10
    particle_number = 10

    # Model

    Bower_stream_function = Bower_StreamFunction(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )
    Bower_stream_function_in_moving_frame = Bower_StreamFunction_in_moving_frame(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )

    Cell_Flow = Cellular_Flow(epsilon=0, L=100, alpha=10, phi=0)

    Inertial_waves = Bower_StreamFunction_with_inertial_waves(
        psi_0=4e3, A=50, L=400, width=40, c_x=10, f=8.64, gamma=0.01
    )

    # Trajectories

    Particles_1 = Evolution_Particles(
        stream_function=Inertial_waves,
        time_step=time_step,
        t_final=t_final,
        particle_number=particle_number,
        x_min=0,
        x_max=2 * Inertial_waves.L,
        y_min=-250,
        y_max=250,
    )

    Particles_2 = Evolution_Particles(
        stream_function=Bower_stream_function,
        time_step=time_step,
        t_final=t_final,
        particle_number=particle_number,
        x_min=0,
        x_max=2 * Bower_stream_function.L,
        y_min=-250,
        y_max=250,
    )

    # Spectral Analysis

    Spec = Spectral_Particles_Analysis(
        time_step=time_step, t_final=t_final, num_particles=particle_number
    )

    u_1, v_1 = Particles_1.velocity_profiles
    u_2, v_2 = Particles_2.velocity_profiles
    Spec.plot_velocity_spectrum(u_2, v_2)
    Spec.plot_velocity_spectrum(u_1, v_1)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
