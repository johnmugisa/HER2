import numpy as np
import pandas as pd
import cmath as cm
from Leastsq import LeastsqMin

# --- For the main procedure and parameters see bottom part of this file

def ZWarburg():
    fmin = 0.01
    fmax = 1e4
    omega = np.zeros(1, dtype=np.float64)
    omega[0] = 2 * np.pi * fmin
    # alpha = 0.2328467394   # --- 11 points per decade
    alpha = 0.1103363182  # --- 22 points per decade
    # alpha = 0.07226722201  # --- 33 points per decade
    for n in range(1000):
        omega = np.append(omega, (1 + alpha) * omega[n])
        if omega[-1] > 2 * np.pi * fmax:
            break
    omega = np.delete(omega, -1)  # --- remove last omega from the array
    omega = np.flip(omega)        # --- reverse order of elements in omega
    print('omega max =', omega[0], ', omega size = ', omega.size)
    tast = 1
    zWarb = np.zeros(omega.size, dtype=np.complex128)
    for k in range(omega.size):
        psi = cm.sqrt(1j * tast * omega[k])
        zWarb[k] = cm.tanh(psi) / psi

    zrew = zWarb.real
    zimw = - zWarb.imag

    return omega, zrew, zimw   # --- omega is in descending order!


def user_data():
    data = pd.read_csv(
        './immfit1.csv',
        sep=',', header=None
    )
    dflat = data.to_numpy()
    omgu = dflat[:, 0] * 2 * np.pi
    zreu = dflat[:, 1]
    zimu = np.abs(dflat[:, 2])
    return omgu, zreu, zimu


# def user_data():
#     data = pd.read_csv(
#         './MyImpedance.dat',
#         sep=',', header=None
#     )
#     dflat = data.to_numpy()
#     omgu = dflat[:, 0] * 2 * np.pi
#     zreu = dflat[:, 1]
#     zimu = dflat[:, 2]
#     return omgu, zreu, zimu


# ---------------------------------------------------
# This part of the code should be changed
# in accordance with a user's problem.
# By default, calculation of Warburg finite--length
# DRT is performed.
# ---------------------------------------------------
#

fname = './results/Warburg_'   # --- Prefix for output files name

lamT0 = 1e-14   # --- Initial guess for Tikhonov reg. parameter
lampg0 = 0.01   # --- Initial guess for PG reg. parameter
mode = 'real'   # --- Use real part of impedance for DRT calculation.
                # --- Set mode = 'imag' to use imaginary part.

omg, zre, zim = ZWarburg()

# --- To supply your data, comment or remove the previous line and
# --- uncomment the next line:
omg, zre, zim = user_data()
# --- The file 'MyImpedance.dat' must be placed
# --- in the directory with python codes. This file
# --- must contain 3 columns separated by commas:
# ---       f(Hz),  zre,  zim
# --- frequencies f must be in descending order and
# --- zim must be positive.
# ---------------------------------------------------
# --- By default, the constrained TRF minimizer is used.
# --- To call Levenberg-Marquardt change the next
# --- line to: keylsq = 'lm'
keylsq = 'lm'
# ---------------------------------------------------

zre -= zre[0]

solver = LeastsqMin(omg, zre, zim, lamT0, lampg0, fname, mode)
solver.driver(keylsq)
# lamvec = [lamT0, lampg0]
# solver = TikhPGSolver(zre, zim, omg, mode, lamT0)
# solution, rpoly, iterations = solver.pg_solver(lamvec)

# solver = TikhPGSolver(zre, zim, omg, mode, lamT0, lampg0)
# solver.driver(keylsq)