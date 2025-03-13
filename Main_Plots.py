from plots import Plots
import time
import numpy as np
from Stream_Functions import (
    Axisymmetric_Vortex,
    Bower_StreamFunction_with_inertial_waves,
)


def main() -> None:
    start_time = time.time()

    # Backup Folder

    filepath_spectrum_Bower = "Plots/Bower_Flow/Spectrum/"
    filepath_stats_Bower = "Plots/Bower_Flow/Stats/"
    savefig = True

    # Parameters

    time_step = 0.01  # One per minute
    t_final = 100
    num_pairs = (50) ** 2
    f = 8.64
    Gamma = np.array([0, 1e-2, 1e-1, 1])
    initial_separation = 1e-1

    # Axisymmetric Vortex

    R = 1
    zeta = 1
    gamma_vortex = 1e-4
    L_vortex = 5

    axisymmetric_vortex = Axisymmetric_Vortex(
        L=L_vortex, R=R, zeta=zeta, gamma=gamma_vortex
    )

    Inertial_Waves = Bower_StreamFunction_with_inertial_waves(
        psi_0=4e3, A=50, L=400, width=40, c_x=10, f=f, gamma=0
    )

    # Class for the plots

    plots = Plots(
        time_step=time_step,
        t_final=t_final,
        num_pairs=num_pairs,
        f=f,
        Gamma=Gamma,
        initial_separation=initial_separation,
        x_min_init=-150,
        x_max_init=150,
        y_min_init=-40,
        y_max_init=40,
    )

    plots.relative_dispersion_gamma(savefig=savefig, filepath=filepath_stats_Bower)
    plots.velocity_spectrum_gamma(savefig=savefig, filepath=filepath_spectrum_Bower)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
