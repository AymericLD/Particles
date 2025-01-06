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
from numpy.typing import NDArray
import matplotlib.pyplot as plt


start_time = time.time()


def velocity_spectrum_gamma(
    psi_0: float,
    A: float,
    L: float,
    width: float,
    c_x: float,
    time_step: float,
    t_final: float,
    particle_number: int,
    f: float,
    Gamma: NDArray,
    savefig: bool,
    filepath: str,
):
    plt.figure(figsize=(10, 6))
    for g in Gamma:
        Inertial_waves = Bower_StreamFunction_with_inertial_waves(
            psi_0=psi_0, A=A, L=L, width=width, c_x=c_x, f=f, gamma=g
        )
        Particles = Evolution_Particles(
            stream_function=Inertial_waves,
            time_step=time_step,
            t_final=t_final,
            particle_number=particle_number,
            x_min=0,
            x_max=2 * Inertial_waves.L,
            y_min=-250,
            y_max=250,
        )
        Spec = Spectral_Particles_Analysis(
            time_step=time_step,
            t_final=t_final,
            num_particles=particle_number,
            f=f,
        )
        u, v = Particles.velocity_profiles
        Spec.plot_velocity_spectrum(u, v, filepath=filepath, savefig=2, f=f, gamma=g)
        # savefig=2 is used for plotting several spectra on the same figure
    plt.xlabel("Frequency (d^-1)")
    plt.ylabel("Spectral Density")
    plt.title("Velocity Spectral Density in frequency with varying gamma")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(
            f"{filepath}Velocity_Spectrum,f={f}gamma={Gamma}num_particles={particle_number}.png"
        )
    else:
        plt.show()


def main() -> None:
    # Backup Folder

    filepath = "Plots/Bower_Flow/Spectrum/"
    savefig = False

    # Parameters

    time_step = 1 / 24 / 60
    t_final = 10
    particle_number = 100
    psi_0 = 4e3
    A = 50
    L = 400
    width = 40
    c_x = 10
    f = 8.64
    gamma = 100

    # Model

    Bower_stream_function = Bower_StreamFunction(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )
    Bower_stream_function_in_moving_frame = Bower_StreamFunction_in_moving_frame(
        psi_0=4e3, A=50, L=400, width=40, c_x=10
    )

    Cell_Flow = Cellular_Flow(epsilon=0, L=100, alpha=10, phi=0)

    Inertial_waves = Bower_StreamFunction_with_inertial_waves(
        psi_0=psi_0, A=A, L=L, width=width, c_x=c_x, f=f, gamma=gamma
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
        time_step=time_step, t_final=t_final, num_particles=particle_number, f=f
    )

    u_1, v_1 = Particles_1.velocity_profiles
    u_2, v_2 = Particles_2.velocity_profiles
    # Spec.plot_velocity_spectrum(
    #     u_2, v_2, filepath=filepath, savefig=1, f=f, gamma=gamma
    # )
    # Spec.plot_velocity_spectrum(
    #     u_1, v_1, filepath=filepath, savefig=1, f=f, gamma=gamma
    # )

    # Spectra for varying gamma

    Gamma = np.array([1e-3, 1e-2, 1e-1, 1])

    velocity_spectrum_gamma(
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
        savefig=savefig,
        filepath=filepath,
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
