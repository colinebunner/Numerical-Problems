# Calculate the power spectrum associated with a correlation function of the form
#	C(\tau) = exp(-|\tau|/\tau_C)
#
# Plot S(\omega) versus ln(\omega). Notice that S(\omega) is fairly flat out to
# \omega ~= 1/\tau_C, at which point it decreases as \omega^{-2}. One calls this
# flat region a "white" region since it contains all frequencies equally. What
# form of C(\tau) would lead to a white spectrum for all frequencies?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.fftpack import fft, ifft

rc('text', usetex=True)

# Critical value of tau from problem statement
tauC = 0.05

# Choose sampling domain and number of samples
tFinal     = 1.0
nSamples   = 500
tau        = np.linspace(0.0, tFinal, num = nSamples)
# Calculate sampled omegas (first half are positive frequencies, second half are
# negative. This is because the DFT treats the signal as periodic.)
omega      = (np.arange(nSamples)*((2*np.pi)/tFinal))[0:nSamples//2]
# Correlation function
expCorr    = np.exp(-tau/tauC)
# Power spectrum of expCorr
powerSpec  = fft(expCorr, n=nSamples)[0:nSamples//2]/nSamples

fig, ax = plt.subplots(1, 2)

ax[0].plot(tau, expCorr)
ax[0].set_xlabel(r"$\tau$ [s]")
ax[0].set_ylabel(r"$C(\tau)$")
ax[1].plot(omega, np.abs(powerSpec))
ax[1].set_xlabel(r"$\omega$ [rad/s]")
ax[1].set_ylabel(r"$|S(\omega)|/N$")

plt.tight_layout()
plt.show()

# Let's answer the second part. The question is essentially for which function
# is there no correlation between the signal and a frequency. One can guess
# that a random function fits this bill. This indeed works (see below), but
# the random function must sample randomly over a symmetric interval about 0,
# otherwise the 0 frequency component will be equal to the mean value over the
# interval.

np.random.seed(2)
expCorr   = np.random.ranf(size=nSamples)-0.5
powerSpec = fft(expCorr, n=nSamples)[0:nSamples//2]/nSamples

fig, ax = plt.subplots(1, 2)

ax[0].plot(tau, expCorr)
ax[1].plot(omega, np.abs(powerSpec))
ax[0].plot(tau, expCorr)
ax[0].set_xlabel(r"$\tau$ [s]")
ax[0].set_ylabel(r"$C(\tau)$")
ax[1].plot(omega, np.abs(powerSpec))
ax[1].set_xlabel(r"$\omega$ [rad/s]")
ax[1].set_ylabel(r"$|S(\omega)|/N$")

plt.tight_layout()
plt.show()
