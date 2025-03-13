from Stream_Functions import (
    StreamFunction,
    Bower_StreamFunction,
    Bower_StreamFunction_in_moving_frame,
    Bower_StreamFunction_with_inertial_waves,
    Cellular_Flow,
    Axisymmetric_Vortex,
)
from Spectral_Analysis import Spectral_Particles_Analysis
from Particles import Evolution_Particles
import time
import numpy as np

start_time = time.time()


def main() -> None:
    # Parameters

    # Bower Flow

    num_pairs = (3) ** 2
    time_step = 1 / 24 / 60
    t_final = 1000
    psi_0 = 4e3
    A = 50
    L = 400
    width = 40
    c_x = 10
    f = 8.64
    Gamma = np.array([0, 1e-3, 1e-2, 1e-1, 1])
    L_cf = 100
    initial_separation = 1e-3

    # Cellular Flow

    epsilon_0 = 4
    epsilon = L / epsilon_0
    alpha = 10
    phi = 0

    # Axisymmetric Vortex

    R = 1
    zeta = 1
    gamma_vortex = 10
    L_vortex = 5

    # Model

    Bower_stream_function = Bower_StreamFunction(
        psi_0=psi_0, A=A, L=L, width=width, c_x=c_x
    )

    Bower_stream_function_in_moving_frame = Bower_StreamFunction_in_moving_frame(
        psi_0=psi_0, A=A, L=L, width=width, c_x=c_x
    )

    Inertial_Waves = Bower_StreamFunction_with_inertial_waves(
        psi_0=psi_0, A=A, L=L, width=width, c_x=c_x, f=f, gamma=Gamma
    )

    cellular_flow = Cellular_Flow(epsilon=epsilon, L=L_cf, alpha=alpha, phi=phi)

    axisymmetric_vortex = Axisymmetric_Vortex(
        L=L_vortex, R=R, zeta=zeta, gamma=gamma_vortex
    )

    # Trajectories

    # Particles = Evolution_Particles(
    #     stream_function=Bower_stream_function,
    #     time_step=time_step,
    #     t_final=t_final,
    #     num_pairs=num_pairs,
    #     x_min_init=-4,
    #     x_max_init=4,
    #     y_min_init=-4,
    #     y_max_init=4,
    #     x_min=-10,
    #     x_max=10,
    #     y_min=-10,
    #     y_max=10,
    #     initial_separation=initial_separation,
    # )

    Particles = Evolution_Particles(
        stream_function=Bower_stream_function_in_moving_frame,
        time_step=time_step,
        t_final=t_final,
        num_pairs=num_pairs,
        x_min_init=-150,
        x_max_init=150,
        y_min_init=-100,
        y_max_init=100,
        x_min=0,
        x_max=2 * Bower_stream_function.L,
        y_min=-250,
        y_max=250,
        initial_separation=initial_separation,
    )

    Particles.plot_trajectories(
        name="Bower", savefig=False, streamlines=True, filepath="Plots/"
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
