# import numpy as np
# from scipy.linalg import lstsq
# from scipy.signal import find_peaks, peak_widths, peak_prominences
# import jax
# import jax.numpy as jnp
# import jaxopt
# from jaxopt import ProjectedGradient
# from jaxopt.objective import ridge_regression

# jax.config.update("jax_enable_x64", True)
# from functools import partial


# # --- Angular frequency omg must be in descending order!
# class TikhPGSolver:   # --- Tikhonov + PG solver.
#     def __init__(self, zexp_re, zexp_im, omg, mode, lamT0):
#         self.rpol = zexp_re[-1] - zexp_re[0]
#         self.zexp_re_norm = zexp_re / self.rpol
#         self.zexp_im_norm = zexp_im / self.rpol
#         self.omg = omg
#         self.mode = mode
#         self.lamT0 = lamT0
#         self.niter = 80
#         self.flagiter = 0

#         self.tau = 1 / self.omg
#         self.lntau = jnp.log(self.tau)
#         self.dlntau = self.create_dmesh(self.lntau)
#         self.dtau = self.create_dmesh(self.tau)
#         self.Idm = jnp.identity(self.omg.size, dtype=jnp.integer)
#         self.am = jnp.zeros((self.omg.size, self.omg.size), dtype=jnp.float64)
#         self.CreateTikhMatrix()
#         self.jacobian = jax.jacobian(self.Tikh_residual)

#     def create_dmesh(self, grid):
#         dh = jnp.zeros(self.omg.size, dtype=jnp.float64)
#         for j in range(1, self.omg.size - 1):
#             dh = dh.at[j].set(0.5 * (grid[j + 1] - grid[j - 1]))
#         dh = dh.at[0].set(0.5 * (grid[1] - grid[0]))
#         dh = dh.at[-1].set(0.5 * (grid[-1] - grid[-2]))
#         return dh

#     def CreateTikhMatrix(self):   # --- creates lhs matrix and rhs vector
#         for i in range(self.omg.size):
#             prod = self.omg[i] * self.tau
#             if self.mode == 'real':
#                 self.am = self.am.at[i, :].set(self.dlntau / (1 + prod**2))
#             else:
#                 self.am = self.am.at[i, :].set(prod * self.dlntau / (1 + prod**2))

#         self.amT = self.am.transpose()                    # --- transposed a-matrix
#         self.amTam = jnp.matmul(self.amT, self.am)
#         self.amTikh = self.amTam + self.lamT0 * self.Idm  # --- Tikhonov matrix

#         if self.mode == 'real':
#             self.brs = jnp.matmul(self.amT, self.zexp_re_norm)    # --- Tikhonov right side vector
#         else:
#             self.brs = jnp.matmul(self.amT, self.zexp_im_norm)    # --- Tikhonov right side vector

#     def Tikh_solver(self, lamt):   # --- solves Tikhonov equation
#         self.amTikh = self.amTam + lamt * self.Idm  # --- new Tikhonov matrix
#         sol, residuals, rank, sv = jnp.linalg.lstsq(self.amTikh, self.brs, rcond=None)
#         return sol


#     # def pg_solver(self, lamvec, norm = 0.0, norm_new = 0.0):
#     #     lamT, lampg = lamvec
#     #     gtau = self.Tikh_solver(lamT)  # --- initial Gfun from Tikhonov solver
#     #     norm, k = 0.0, 0
#     #     for k in range(self.niter * 1000):   # --- PG-iterations:
#     #         gk = gtau - lampg * (jnp.matmul(self.amTikh, gtau) - self.brs)
#     #         gtau = gk.clip(0.0)                          # --- replace negatives by 0
#     #         if k % 10 == 0:
#     #             norm_new = jnp.sqrt(jnp.sum(gtau**2))
#     #             if jnp.abs(norm_new - norm) / norm_new < 1e-8:
#     #                 break
#     #             norm = norm_new
#     #     rpoly = jnp.sum(gtau * self.dlntau)   # --- dimensionless pol. resistance, should be 1
#     #     if k == self.niter * 1000 - 1:
#     #         self.flagiter = 1
#     #     return gtau, rpoly, k

#     def pg_solver(self, lamvec):

#         lamT, lampg = lamvec  # these are the two regularization parameters
#         gtau = self.Tikh_solver(lamT)  # --- initial Gfun from Tikhonov solver
#         norm, k = 0.0, 0
#         for k in range(self.niter * 1000):   # --- PG-iterations:
#             gk = gtau - lampg * (jnp.matmul(self.amTikh, gtau) - self.brs)
#             gtau = gk.clip(0.0)                          # --- replace negatives by 0
#             if k % 10 == 0:
#                 norm_new = jnp.sqrt(jnp.sum(gtau**2))
#                 if jnp.abs(norm_new - norm) / norm_new < 1e-8:
#                     break
#                 norm = norm_new
#         rpoly = np.sum(gtau * self.dlntau)   # --- dimensionless pol. resistance, should be 1
#         if k == self.niter * 1000 - 1:
#             self.flagiter = 1
#         return gtau, rpoly, k



#     def jacoby(self, pvec):
#         return (jax.jacobian(self.Tikh_residual)(jnp.array(pvec)))

#     def Tikh_residual(self, lamvec):   # --- returns vector of Tikhonov residuals to be minimized
#         gfvec, rp, kk = self.pg_solver(lamvec)   # --- (new amTikh has been calculated)
#         resid = jnp.matmul(self.amTikh, gfvec) - self.brs
#         return resid

#     def Tikh_residual_norm(self, gtau, lamT):    # --- returns mean Tikhonov residual
#         self.amTikh = self.amTam + lamT * self.Idm    # --- Tikhonov matrix
#         work = jnp.matmul(self.amTikh, gtau)
#         sumres = jnp.sqrt(jnp.sum((work - self.brs)**2))
#         sumlhs = jnp.sqrt(jnp.sum(work**2))
#         return sumres, sumlhs

#     def residual_norm(self, gtau):   # --- returns residual
#         work = jnp.matmul(self.amTam, gtau)
#         normres = jnp.sqrt(jnp.sum((work - self.brs)**2))
#         return normres

#     def Zmodel_imre(self, gtau):   # --- calculates model Im(Z)/Re(Z)
#         zmod = jnp.zeros(self.omg.size, dtype=jnp.float64)
#         for i in range(self.omg.size):
#             prod = self.omg[i] * self.tau
#             if self.mode == 'real':
#                 integrand = gtau / (1 + prod ** 2)
#             else:
#                 integrand = prod * gtau / (1 + prod ** 2)
#             zmod = zmod.at[i].set(jnp.sum(self.dlntau * integrand))  # --- my trapezoid
#         return jnp.flip(self.rpol * zmod)

#     def rpol_peaks(self, gtau):   # --- finds and integrates gamma-peaks. Beta version.
#         print(gtau)
#         peaks, dummy = find_peaks(np.asarray(gtau.copy()), prominence=0.01)
#         width = peak_widths(gtau, peaks, rel_height=1)

#         integr = jnp.zeros(peaks.size, dtype=jnp.float64)
#         for n in range(peaks.size):
#             lb, ub = int(width[2][n]), int(width[3][n])
#             integr = integr.at[n].set(jnp.sum(gtau[lb:ub] * self.dlntau[lb:ub]))

#         pparms = jnp.zeros((2, peaks.size), dtype=jnp.float64)
#         pparms = pparms.at[0, :].set(jnp.flip(1 / (2 * jnp.pi * self.tau[peaks])))   # --- peak frequencies
#         pparms = pparms.at[1, :].set(jnp.flip(integr))                   # --- peak polarization fractions
#         return pparms

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jaxopt
from jaxopt import ProjectedGradient
from scipy.signal import find_peaks, peak_widths
from jaxopt.objective import ridge_regression
jax.config.update("jax_enable_x64", True)
import cmath as cm


class TikhPGSolver:
    def __init__(self, zexp_re, zexp_im, omg, mode, lamT0):
        self.rpol = zexp_re[-1] - zexp_re[0]
        self.zexp_re_norm = zexp_re / self.rpol
        self.zexp_im_norm = zexp_im / self.rpol
        self.omg = omg
        self.mode = mode
        self.lamT0 = lamT0
        self.niter = 80
        self.flagiter = 0

        self.tau = 1 / self.omg
        self.lntau = jnp.log(self.tau)
        self.dlntau = self.create_dmesh(self.lntau)
        self.dtau = self.create_dmesh(self.tau)
        self.Idm = jnp.identity(self.omg.size, dtype=jnp.integer)
        self.am = jnp.zeros((self.omg.size, self.omg.size), dtype=jnp.float64)
        self.CreateTikhMatrix()
        self.jacobian = jax.jacobian(self.Tikh_residual)

    def create_dmesh(self, grid):
        dh = jnp.zeros(self.omg.size, dtype=jnp.float64)
        for j in range(1, self.omg.size - 1):
            dh = dh.at[j].set(0.5 * (grid[j + 1] - grid[j - 1]))
        dh = dh.at[0].set(0.5 * (grid[1] - grid[0]))
        dh = dh.at[-1].set(0.5 * (grid[-1] - grid[-2]))
        return dh

    def CreateTikhMatrix(self):   # --- creates lhs matrix and rhs vector
        for i in range(self.omg.size):
            prod = self.omg[i] * self.tau
            if self.mode == 'real':
                self.am = self.am.at[i, :].set(self.dlntau / (1 + prod**2))
            else:
                self.am = self.am.at[i, :].set(prod * self.dlntau / (1 + prod**2))

        self.amT = self.am.transpose()                    # --- transposed a-matrix
        self.amTam = jnp.matmul(self.amT, self.am)
        self.amTikh = self.amTam + self.lamT0 * self.Idm  # --- Tikhonov matrix

        if self.mode == 'real':
            self.brs = jnp.matmul(self.amT, self.zexp_re_norm)    # --- Tikhonov right side vector
        else:
            self.brs = jnp.matmul(self.amT, self.zexp_im_norm)    # --- Tikhonov right side vector

    def Tikh_solver(self, lamt):
        self.amTikh = self.amTam + lamt * self.Idm  # --- new Tikhonov matrix
        sol = jnp.linalg.solve(self.amTikh, self.brs)  # --- Solve Tikhonov equation
        return sol



        return sol

    def pg_solver(self, lamvec):
        lamT, lampg = lamvec  # these are the two regularization parameters
        gtau = self.Tikh_solver(lamT)  # --- initial Gfun from Tikhonov solver

        objective_fun = lambda gtau: jnp.sum((jnp.matmul(self.amTikh, gtau) - self.brs) ** 2)
        projection = lambda gtau, _: jnp.clip(gtau, 0.0, None)

        pg = ProjectedGradient(fun=objective_fun, projection=projection, maxiter=self.niter * 1000)
        solution = pg.run(gtau, lampg)


        rpoly = np.sum(solution.params * self.dlntau)

        return solution.params, rpoly, solution.state.iter_num



    def jacoby(self, pvec):
        return (jax.jacobian(self.Tikh_residual)(jnp.array(pvec)))

    def Tikh_residual(self, lamvec):   # --- returns vector of Tikhonov residuals to be minimized
        gfvec, rp, kk = self.pg_solver(lamvec)   # --- (new amTikh has been calculated)
        resid = jnp.matmul(self.amTikh, gfvec) - self.brs
        return resid

    def Tikh_residual_norm(self, gtau, lamT):    # --- returns mean Tikhonov residual
        self.amTikh = self.amTam + lamT * self.Idm    # --- Tikhonov matrix
        work = jnp.matmul(self.amTikh, gtau)
        sumres = jnp.sqrt(jnp.sum((work - self.brs)**2))
        sumlhs = jnp.sqrt(jnp.sum(work**2))
        return sumres, sumlhs

    def residual_norm(self, gtau):   # --- returns residual
        work = jnp.matmul(self.amTam, gtau)
        normres = jnp.sqrt(jnp.sum((work - self.brs)**2))
        return normres

    def Zmodel_imre(self, gtau):   # --- calculates model Im(Z)/Re(Z)
        zmod = jnp.zeros(self.omg.size, dtype=jnp.float64)
        for i in range(self.omg.size):
            prod = self.omg[i] * self.tau
            if self.mode == 'real':
                integrand = gtau / (1 + prod ** 2)
            else:
                integrand = prod * gtau / (1 + prod ** 2)
            zmod = zmod.at[i].set(jnp.sum(self.dlntau * integrand))  # --- my trapezoid
        return jnp.flip(self.rpol * zmod)

    def rpol_peaks(self, gtau):
        gtau_numpy = np.array(np.asarray(gtau))  # converting gtau to a writable numpy array
        peaks, dummy = find_peaks(gtau_numpy, prominence=0.01)
        width = peak_widths(gtau_numpy, peaks, rel_height=1)

        integr = jnp.zeros(peaks.size, dtype=jnp.float64)
        for n in range(peaks.size):
            lb, ub = int(width[2][n]), int(width[3][n])
            integr = integr.at[n].set(jnp.sum(gtau[lb:ub] * self.dlntau[lb:ub]))

        pparms = jnp.zeros((2, peaks.size), dtype=jnp.float64)
        pparms = pparms.at[0, :].set(jnp.flip(1 / (2 * jnp.pi * self.tau[peaks])))   # --- peak frequencies
        pparms = pparms.at[1, :].set(jnp.flip(integr))                   # --- peak polarization fractions
        return pparms

