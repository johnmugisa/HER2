import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

xstring = 'Re(Z) / Ohm cm$^2$'
ystring = '-Im(Z) / Ohm cm$^2$'

class Plotter:
    def __init__(self, zexp_re, zexp_im, omg, mode):
        self.zexp_re = zexp_re
        self.zexp_im = zexp_im
        self.omg = omg
        self.mode = mode
        if mode == 'real':
            self.zexp_prt = np.flip(self.zexp_re)
            self.ylabel = 'Re(Z) / Ohm cm^2'
        else:
            self.zexp_prt = np.flip(self.zexp_im)
            self.ylabel = '-Im(Z) / Ohm cm^2'
        self.fHz = np.flip(omg / (2 * np.pi))
        self.fmin = self.floor_power_of_10(np.min(self.fHz))
        self.fmax = self.ceil_power_of_10(np.max(self.fHz))
        plt.rcParams.update({'font.size': 18})

    def ceil_power_of_10(self, x):
        logx = np.log10(x)
        pow = np.ceil(logx)
        return 10**pow

    def floor_power_of_10(self, x):
        logx = np.log10(x)
        pow = np.floor(logx)
        return 10**pow

    def plotgamma_hires(self, gamma_tau, fname, lab):
        gammaf = np.flip(gamma_tau)
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=70)
        plt.tight_layout()
        plt.scatter(self.fHz, gammaf, color='blue', marker='.')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        # plt.plot(self.fHz, gammaf, color='blue', lw=1)
        ymin = np.min(gammaf)
        ymax = np.max(gammaf)
        plt.axis([self.fmin, self.fmax, 1.2 * ymin, 1.2 * ymax])
        plt.xscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel('DRT / 1/s')
        plt.xlabel('Frequency / Hz')
        plt.savefig(fname + '.pdf')

    def plotgamma(self, gamma_tau, fname, keysave, lab):
        gammaf = np.flip(gamma_tau)
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=70)
        plt.tight_layout()
        plt.scatter(self.fHz, gammaf, marker='.')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        # plt.plot(self.fHz, gammaf, color='blue', lw=1)
        ymin = np.min(gammaf)
        ymax = np.max(gammaf)
        plt.axis([self.fmin, self.fmax, 1.2 * ymin, 1.2 * ymax])
        plt.xscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel('DRT / 1/s')
        plt.xlabel('Frequency / Hz')
        if keysave == 1:
            self.plotgamma_hires(gamma_tau, fname, lab)
        plt.show()

    def plotGfun(self, gfun, fname, keysave, lab):
        Gfun = np.flip(gfun)
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=70)
        plt.tight_layout()
        plt.scatter(self.fHz, Gfun, marker='.', color='blue')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        plt.plot(self.fHz, Gfun, color='blue', lw=1)
        ymin = 0.0
        ymax = np.max(Gfun)
        plt.axis([self.fmin, self.fmax, 1.2 * ymin, 1.2 * ymax])
        plt.xscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel('dimensionless DRT')
        plt.xlabel('Frequency / Hz')
        if keysave == 1:
            np.savetxt(fname + '.dat',
                       np.transpose([self.fHz, Gfun]), fmt="%f")
            self.plotGfun_hires(gfun, fname, lab)
        plt.show()

    def plotGfun_hires(self, gfun, fname, lab):
        Gfun = np.flip(gfun)
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=300)
        plt.tight_layout()
        plt.scatter(self.fHz, Gfun, marker='.', color='blue')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        plt.plot(self.fHz, Gfun, color='blue', lw=1)
        ymin = 0.0
        ymax = np.max(Gfun)
        plt.axis([self.fmin, self.fmax, 1.2 * ymin, 1.2 * ymax])
        plt.xscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel('dimensionless DRT')
        plt.xlabel('Frequency / Hz')
        plt.savefig(fname + '.pdf')

    def plot_Z_hires(self, zmodel, fname, lab):
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=300)
        plt.tight_layout()
        plt.scatter(self.fHz, self.zexp_prt, color='blue', marker='.')
        plt.scatter(self.fHz, zmodel, s=30, facecolors='none', edgecolors='r')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        ymin = 0
        ymax = np.max(self.zexp_prt)
        plt.axis([self.fmin, self.fmax, ymin, 1.2 * ymax])
        plt.xscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel(self.ylabel)
        plt.xlabel('Frequency / Hz')
        plt.savefig(fname + '.pdf')

    def plotshow_Z(self, zmodel, fname, keysave, lab):
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=70)
        plt.tight_layout()
        plt.scatter(self.fHz, self.zexp_prt, color='blue', marker='.')
        plt.scatter(self.fHz, zmodel, s=30, facecolors='none', edgecolors='r')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        ymin = 0
        ymax = np.max(self.zexp_prt)
        plt.axis([self.fmin, self.fmax, ymin, 1.2 * ymax])
        plt.xscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel(self.ylabel)
        plt.xlabel('Frequency / Hz')
        if keysave == 1:
            np.savetxt(fname + '.dat',
                       np.transpose([self.fHz, self.zexp_prt, zmodel]), fmt="%f")
            self.plot_Z_hires(zmodel, fname,lab)
        plt.show()

    def plotNyq(self, zmodel, lab):
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=70)
        plt.tight_layout()
        plt.scatter(self.zexp_re, self.zexp_im, marker='.')
        plt.scatter(zmodel.real, - zmodel.imag, s=30, facecolors='none', edgecolors='r')
        plt.annotate(lab, xy=(0.6, 0.9), xycoords='axes fraction')
        xmin = np.min(np.concatenate([self.zexp_re, zmodel.real]))
        xmax = np.max(np.concatenate([self.zexp_re, zmodel.real]))
        plt.axis([1.05 * xmin, 1.05 * xmax, 0, 1.05 * (xmax - xmin)])
        plt.ylabel(ystring)
        plt.xlabel(xstring)
        plt.show()

    def plotLambda(self, resid, solnorm, arrlam, fname, lab, ksave):
        plt.figure(num=1, figsize=(8.5, 6.5), dpi=70)
        plt.tight_layout()
        plt.scatter(resid, solnorm, marker='+')
        plt.annotate(lab, xy=(0.9, 0.9), xycoords='axes fraction')
        ymin = self.floor_power_of_10(np.min(solnorm))
        ymax = self.ceil_power_of_10(np.max(solnorm))
        xmin = self.floor_power_of_10(np.min(resid))
        xmax = self.ceil_power_of_10(np.max(resid))
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')
        plt.ylabel('solution norm ||$x$||')
        plt.xlabel('residual ||$A x - b$||')
        plt.rcParams.update({'font.size': 8})
        for a, b, c in zip(resid, solnorm, arrlam):
            plt.text(a, b, "{:.0e}".format(c))
        if ksave == 1:
            plt.savefig(fname + '.pdf')
        plt.rcParams.update({'font.size': 18})
        plt.show()
