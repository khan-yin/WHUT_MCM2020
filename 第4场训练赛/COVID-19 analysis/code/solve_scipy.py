from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

from common import *

def lorenz(w, t, p, r, b):
    # 给出位置矢量w，和三个参数p, r, b计算出
    # dx/dt, dy/dt, dz/dt的值
    x, y, z = w
    # 直接与lorenz的计算公式对应
    return np.array([p*(y-x), x*(r-z)-y, x*y-b*z])

def test_lorentz():
    t = np.arange(0, 30, 0.01) # 创建时间点
    # 调用ode对lorenz进行求解, 用两个不同的初始值
    track1 = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))
    track2 = odeint(lorenz, (0.0, 1.01, 0.0), t, args=(10.0, 28.0, 3.0))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(track1[:,0], track1[:,1], track1[:,2])
    ax.plot(track2[:,0], track2[:,1], track2[:,2])
    plt.show()

def f1(u, t,
       beta, start, hStart,
       k, epsilon, lamb, theta, delta, sigma, gamma, eta):
    S, E, I, Q, J, C, R, D = u

    h = 0.9*(t > hStart)
    # beta = (31+t)/(22+5*t)
    return np.array([
        -beta*(k*E+I)*(1-h)*S,
        beta*(k*E+I)*(1-h)*S - (epsilon+lamb)*E,
        epsilon*E - (theta+delta)*I,
        lamb*E - (sigma+delta)*Q,
        theta*I + sigma*Q - (gamma+delta)*J,
        theta * (I+Q) + sigma * Q,
        gamma*J,
        delta*I + eta*J
    ])*(t > start)

def f2(u, t,
       beta, start, hStart,
       k, epsilon, lamb, theta, delta, sigma,
       gamma0, gamma1, mStart, mEnd,
       eta):
    S, E, I, Q, J, C, R, D = u

    h = 0.9*(t > hStart)
    if t < mStart:
        gamma = gamma0
    elif t > mEnd:
        gamma = gamma1
    else:# 在时间介于mStart和mEnd之间时，gamma线性增加，表示医疗资源扩充或者引进医疗技术。
        gamma = gamma0 + (t-mStart)*(gamma1-gamma0)/(mEnd-mStart)

    dS = -beta*(k*E+I)*(1-h)*S
    dE = beta*(k*E+I)*(1-h)*S - (epsilon+lamb)*E
    dI = epsilon*E - (theta+delta)*I
    dQ = lamb*E - sigma*Q
    dJ = theta*I + sigma*Q - (gamma+delta)*J
    dC = theta * I + sigma * Q
    dR = gamma*J
    dD = delta*I + eta*J
    return np.array([dS, dE, dI, dQ, dJ, dC, dR, dD])*(t > start)


def test_SEIR(province, p=None, save=True):
    if p is None:
        p = bestParametersDict[province]

    u0 = (p["S0"], 109, p["I0"], p["Q0"], 0, 0, 0, 0)

    parameters = (p["betaTimesS0"]/p["S0"], p["start"], p["hStart"], p["k"], p["epsilon"], p["lamb"],
                  p["theta"], p["delta"], p["sigma"], p["gamma0"], p["gamma1"],
                  p["mStart"], p["mEnd"], p["eta"])
    t = np.arange(0, 200, 1) # 创建时间点
    # 调用ode对lorenz进行求解, 用两个不同的初始值
    track1 = odeint(f2, u0, t, args=parameters)
    r, c = 2, 4
    fig, axs = plt.subplots(r, c, figsize=(3*c, 3*r))
    titles = ["S", "E", "I", "Q", "J", "C", "R", "D"]
    for i in range(r):
        for j in range(c):
            track = track1[:, c*i+j]
            axs[i, j].plot(t, track)
            axs[i, j].set(title="{}".format(titles[c*i+j]))
    if save:
        pdf = PdfPages("..//img//{}_SEIR.pdf".format(province))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()


if __name__ == '__main__':
    test_SEIR("Anhui")
    test_SEIR("Channel Islands")
