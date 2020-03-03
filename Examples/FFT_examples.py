import matplotlib.pyplot as plt
import numpy as np

# Chapter 5 of Applied Linear Algebra by Olver and Shakiban uses the
# function
#       f(x) = 2*pi*x - x^2
# to demonstrate construction of discrete Fourier representations. Here,
# we will reproduce the n = 4 and n = 16 approximations shown in the text,
# both using the noisy high frequencies and the more correct frequency
# wrapping. We will also construct the n = 17 polynomial interpolants to
# demonstrate the subtle difference between the algorithms for even and
# odd number of interpolant points
#
# Feel free to play with f, but always remember not to include identical
# points (i.e. don't sample your function for more than 1 period)

def f(x):
    return np.sin(x)
    #return 2*np.pi*x - x**2.

# For discrete representation of interpolating polynomial
npts = 100

# Number of interpolation points
numInterpolants = [4, 16, 17]

x_fs = np.linspace(0, 2*np.pi, npts)

for n in numInterpolants:
    # The reason I construct x this way is otherwise y[0] == y[-1] (=0),
    # and this causes problems with the polynomial interpolant
    x = np.linspace(0, 2*np.pi, n+1)[0:n]
    y = f(x)
    y_fft = np.fft.fft(y)

    # Naive, noisy interpolating polynomial (Hifreq)
    y_fs1 = np.zeros(npts)
    for i in range(n):
        y_fs1 += 1./n * (
            y_fft[i].real * np.cos(i*x_fs) - y_fft[i].imag * np.sin(i*x_fs)
        )
    
    # Properly constructed interpolating polynomial (Lofreq)
    y_fs2 = np.zeros(npts)
    # zero-frequency term
    zf_term = y_fft[0] * 1./n
    
    if n % 2 == 0:
        upper = int(n//2)
    else:
        upper = int(n//2) + 1
        
    coeffs = y_fft[1:upper] * 2./n

    for i in range(1, len(coeffs) + 1):
        y_fs2 += coeffs[i-1].real * np.cos(i*x_fs) - coeffs[i-1].imag * np.sin(i*x_fs)
    
    y_fs2 += zf_term.real

    # Highest negative frequency contributes additional cos term
    if n % 2 == 0:
        y_fs2 += (1/n) * y_fft[upper].real * np.cos(upper * x_fs)

    fig = plt.figure()
    ax  = fig.gca()

    ax.plot(x_fs, f(x_fs), color='black', label='f(x)')
    ax.plot(x_fs, y_fs1, color='xkcd:cyan', label='hifreq')
    ax.plot(x_fs, y_fs2, color='xkcd:rust red', label='lofreq')
    ax.legend()
    plt.savefig("sinx-{}.png".format(n))
    plt.close(fig)
