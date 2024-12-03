from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
)
from Spectral_Analysis import Spectral_Particles_Analysis
from Particles import Evolution_Particles
import time

start_time = time.time()


def main() -> None:
    # Parameters

    num_particles = 10
    time_step = 1 / 8
    t_final = 10

    # Model

    Bower_stream_function = Bower_StreamFunction(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )

    Bower_stream_function_in_moving_frame = Bower_StreamFunction_in_moving_frame(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )

    # Trajectories

    Particles = Evolution_Particles(
        stream_function=Bower_stream_function,
        time_step=time_step,
        t_final=t_final,
        particle_number=num_particles,
        x_min=0,
        x_max=2 * Bower_stream_function.L,
        y_min=-250,
        y_max=250,
    )

    Particles.plot_trajectories(
        name="Bower", savefig=True, streamlines=False, filepath="Plots/"
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
