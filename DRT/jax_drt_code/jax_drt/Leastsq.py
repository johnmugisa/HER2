import numpy as np
import time
from scipy.optimize import least_squares
from scipy.linalg import norm
import jax
import jax.numpy as jnp
import jaxopt
jax.config.update("jax_enable_x64", True)

from TpgSolver import TikhPGSolver
from PlotsSolver import Plotter


class LeastsqMin:   # --- finds optimal lamT, lampg for TPG-iterations and returns solutions
    def __init__(self, omg, zre, zim, lamT0, lampg0, fname, mode='real'):
        self.omg = omg
        self.zexp_re = zre
        self.zexp_im = zim
        self.lamT0 = lamT0
        self.lampg0 = lampg0
        self.fname = fname
        self.mode = mode
        if mode == 'real':
            self.fsuffix = 'zre'
        else:
            self.fsuffix = 'zim'
        self.myTPG = TikhPGSolver(self.zexp_re, self.zexp_im, self.omg, self.mode, lamT0)
        self.gfun_init = self.myTPG.Tikh_solver(self.lamT0)
        self.fHz = self.omg / (2 * jnp.pi)

    def find_lambda(self):
        kmax, lam1 = 25, 1e-25
        solnorm = jnp.zeros(kmax, dtype=jnp.float64)
        resid = jnp.zeros(kmax, dtype=jnp.float64)
        lamT = jnp.zeros(kmax, dtype=jnp.float64)
        lampg = jnp.zeros(kmax, dtype=jnp.float64)
        for k in range(kmax):
            lam1 = lam1 * 10
            lamT = lamT.at[k].set(lam1)
            gfun = self.myTPG.Tikh_solver(lam1)
            resid = resid.at[k].set(self.myTPG.residual_norm(gfun))
            solnorm = solnorm.at[k].set(jnp.sqrt(jnp.sum(gfun**2)))
            lampg = lampg.at[k].set(1 / norm(self.myTPG.amTikh))
        return resid, solnorm, lamT, lampg

    def driver(self, lsq):   # --- omega must be in descending order!
        myplots = Plotter(self.zexp_re, self.zexp_im, self.omg, self.mode)

        resid, solnorm, arrlamT, arrlampg = self.find_lambda()
        myplots.plotLambda(resid, solnorm, arrlamT,
                           self.fname + '_lambda_T','$\lambda_T^0$', 0)
        myplots.plotLambda(resid, solnorm, arrlampg,
                           self.fname + '_lambda_PG', '$\lambda_{pg}^0$', 0)

        myplots.plotNyq(self.zexp_re - 1j * self.zexp_im, 'Initial spectrum')
        myplots.plotgamma(self.gfun_init, self.fname + '_init', 0, 'Tikhonov gamma')
        zmod = self.myTPG.Zmodel_imre(self.gfun_init)
        myplots.plotshow_Z(zmod, self.fname + '_init' + self.fsuffix, 0, 'Tikhonov solution')

        start = time.time()
        lamvecinit = jnp.array([self.lamT0, self.lampg0], dtype=jnp.float64)
        low, high = lamvecinit / 10, lamvecinit * 10

        if lsq == 'lm':
            resparm = least_squares(self.myTPG.Tikh_residual, lamvecinit,
                                    method='lm', args=())

        else:
            resparm = least_squares(jax.jit(self.myTPG.Tikh_residual), lamvecinit, bounds=(low, high),
                                     args=())
        # if lsq == 'lm':
        #     resparm = least_squares(self.myTPG.Tikh_residual, lamvecinit,
        #                              method='lm', args=())

        # else:
        #     resparm = least_squares(jax.jit(self.myTPG.Tikh_residual), lamvecinit, bounds=(low, high),
        #                             jac=self.myTPG.jacobian(), args=())
        res = resparm.x

        print(f"resparm.x = {res}")

        gfun, rpoly, nit = self.myTPG.pg_solver(res)
        # gamres = 2 * np.pi * self.fHz * gfun
        end = time.time()
        print('Projected gradient iterations =', nit, ', rpol =', rpoly,
              ', Rpol = ', self.myTPG.rpol)
        print('lamTfit, lampgfit =', res, ', elapsed: ', (end - start), ' sec')

        resinit, lhsinit = self.myTPG.Tikh_residual_norm(self.gfun_init, self.lamT0)
        resfin , lhsfin  = self.myTPG.Tikh_residual_norm(gfun, res[0])
        print('Tikhonov residual: initial, final = ', resinit, resfin)
        print('Tikhonov lhs norm: initial, final =', lhsinit, lhsfin)
        if resparm.status > 0 :
            print('Number of Jacobian evaluations =', resparm.njev, ', status = OK')
        if self.myTPG.flagiter == 1:
            print('Warning, limiting number of iterations is achieved')

        # myplots.plotgamma(gamres, self.fname + 'gamma', 1, '')
        myplots.plotGfun(gfun, self.fname + 'Gfun', 1, 'Final G-function' )
        zmod = self.myTPG.Zmodel_imre(gfun)
        myplots.plotshow_Z(zmod, self.fname + self.fsuffix, 1, '')
        peakparms = self.myTPG.rpol_peaks(gfun)

        print('Peak frequencies (beta):   ', ''.join(['{:.5f}  '.format(item) for item in peakparms[0]]))
        print('Peak polarizations (beta): ', ''.join(['{:.5f}  '.format(item) for item in peakparms[1]]))
