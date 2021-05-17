import numpy as np
from diffeqpy import de
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(u, p, t):
    x, y, z = u
    sigma, rho, beta = p
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def f1(u, p, t):
    E, I, Q, J, R = u
    beta, k, epsilon, lamb, theta, delta, sigma, gamma = p
    return [
        beta*(k*E+I) - (epsilon+lamb)*E,
        epsilon*E - (theta+delta)*I,
        lamb*E - sigma*Q,
        theta*I + sigma*Q - (gamma+delta)*J,
        gamma*J
    ]

def test_lorentz():
    u0 = [1.0, 0.0, 0.0]
    tspan = (0., 10.)
    p = [10.0, 28.0, 8/3]
    prob = de.ODEProblem(f, u0, tspan, p)
    sol = de.solve(prob, saveat=0.1)

    ut = np.transpose(sol.u)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(sol.t, sol.u)
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(ut[0, :], ut[1, :], ut[2, :])
    plt.show()

def test_SEIR():
    tspan = (0., 100.)
    u0 = [109, 20, 33, 5, 7]
    p = [5, 0.1, 2/35, 3/35, 1/3, 1/3, 1/17, 1/(17*12)]
    prob = de.ODEProblem(f1, u0, tspan, p)
    sol = de.solve(prob, saveat=0.01)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.R)


if __name__ == '__main__':
    test_SEIR()

