from plots import Plots
import time
import numpy as np


def main() -> None:
    start_time = time.time()

    # Backup Folder

    filepath_spectrum_Bower = "Plots/Bower_Flow/Spectrum/"
    filepath_stats_Bower = "Plots/Bower_Flow/Stats"
    savefig = False

    # Parameters

    time_step = 1 / 1440  # One per minute
    t_final = 10
    particle_number = 100
    psi_0 = 4e3
    A = 50
    L = 400
    width = 40
    c_x = 10
    f = 8.64
    Gamma = np.array([1e-3, 1e-2, 1e-1, 1])
    initial_separation = 5

    # Class for the plots

    plots = Plots(
        psi_0=psi_0,
        A=A,
        L=L,
        width=width,
        c_x=c_x,
        time_step=time_step,
        t_final=t_final,
        particle_number=particle_number,
        f=f,
        Gamma=Gamma,
        initial_separation=initial_separation,
    )

    plots.relative_dispersion_gamma(savefig=savefig, filepath=filepath_stats_Bower)
    plots.velocity_spectrum_gamma(savefig=savefig, filepath=filepath_spectrum_Bower)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
