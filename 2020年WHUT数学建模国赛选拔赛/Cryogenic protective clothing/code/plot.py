from matplotlib.backends.backend_pdf import PdfPages
from common import *
from heat_pde import *


def plot_T_surface(T, save=False, name=""):
    tArray = np.linspace(0, T.shape[0], T.shape[0])
    xArray = np.linspace(0, T.shape[1], T.shape[1])
    tGrids, xGrids = np.meshgrid(tArray, xArray)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surface = ax.plot_surface(xGrids, tGrids, T.T, cmap=cm.coolwarm)
    ax.set_xlabel("$x$", fontdict={"size": 18})
    ax.set_ylabel(r"$\tau$", fontdict={"size": 18})
    ax.set_zlabel(r"$U$", fontdict={"size": 18})
    ax.set_title(u"热传导方程 $u_\\tau = u_{xx}$")
    fig.colorbar(surface, shrink=0.75)
    if save:
        pdf = PdfPages("..//img//T_surface_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_T_slice(T, emp, deltaT=200, save=True, name=""):
    fig = plt.figure()
    for t in range(0, T.shape[0], deltaT):
        plt.plot(range(len(T[t])), T[t], label="第{}秒".format(t))

    if emp is not None:
        plt.scatter([20], [15], label="{:.2f}秒".format(emp[0]))
        plt.scatter([20], [10], label="{:.2f}秒".format(emp[1]))
    plt.xlabel("位置")
    plt.ylabel("温度(摄氏度)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//T_slice{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_x_slice(T, LEN, emp, save=True, name=""):
    fig = plt.figure()
    labels = ["隔热层-功能层", "功能层-织物层", "织物层-体表空气", "体表空气-皮肤", ]
    for i, xi in enumerate(LEN[:-1]):
        plt.plot(range(len(T[:, xi])), T[:, xi], label="{}截面".format(labels[i]))

    if emp is not None:
        plt.scatter([emp[0]], [15], label="工作困难".format())
        plt.scatter([emp[1]], [10], label="生命危险".format())
    plt.xlabel("时间(秒)")
    plt.ylabel("温度(摄氏度)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//x_slice{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_prob1_th_air_sensitivity(th_air_sensitivity15=None,
                                   th_air_sensitivity10=None,
                                   th_airs=[0.6, 0.7, 0.8],
                                   save=True, name=""):
    if th_air_sensitivity10 is None:
        th_air_sensitivity15 = []
        th_air_sensitivity10 = []
        for th_air in th_airs:
            he = HeatEquation(1600, th_air=th_air)
            he.solve()
            th_air_sensitivity15.append(he.time_before(15))
            th_air_sensitivity10.append(he.time_before(10))
        print(th_air_sensitivity15)
        print(th_air_sensitivity10)
    plt.figure()
    plt.plot(th_airs, th_air_sensitivity15, label="体表到达15度", marker='s')
    plt.plot(th_airs, th_air_sensitivity10, label="体表到达10度", marker='s')
    plt.xlabel("空气层厚度(毫米)")
    plt.ylabel("时间(秒)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//th_air_sensitivity{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_prob1_th_skin_sensitivity(th_skin_sensitivity15=None,
                                   th_skin_sensitivity10=None,
                                   th_skins=[0.35, 0.4, 0.45, 0.5],
                                   steps=[2100, 1800, 1700, 1600],
                                   save=True, name=""):
    if th_skin_sensitivity15 is None:
        # [542.04, 525.6, 526.34, 531.62]
        # [2037.1200000000001, 1717.04, 1629.82, 1575.88]
        th_skin_sensitivity15 = []
        th_skin_sensitivity10 = []
        for i, th_air in enumerate(th_skins):
            he = HeatEquation(steps[i], th_air=th_air)
            he.solve()
            th_skin_sensitivity15.append(he.time_before(15))
            th_skin_sensitivity10.append(he.time_before(10))
        print(th_skin_sensitivity15)
        print(th_skin_sensitivity10)
    plt.figure()
    plt.plot(th_skins, th_skin_sensitivity15, label="体表到达15度", marker='s')
    plt.plot(th_skins, th_skin_sensitivity10, label="体表到达10度", marker='s')
    plt.xlabel("皮肤层厚度(毫米)")
    plt.ylabel("时间(秒)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//th_skin_sensitivity{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_prob2_hout_sensitivity(sensitivity15=None,
                                   sensitivity10=None,
                                   args=[3.8, 3.9, 4, 4.1, 4.2],
                                   save=True, name="prob2"):
    if sensitivity15 is None:
        sensitivity15 = []
        sensitivity10 = []
        for arg in args:
            he = HeatEquation(1700, hout=arg*2)
            he.solve()
            sensitivity15.append(he.time_before(15))
            sensitivity10.append(he.time_before(10))
        print(sensitivity15)
        print(sensitivity10)
    plt.figure()
    plt.plot(args, sensitivity15, label="体表到达15度", marker='s')
    plt.plot(args, sensitivity10, label="体表到达10度", marker='s')
    plt.xlabel("外层对流换热系数")
    plt.ylabel("时间(秒)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//hout_sensitivity_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def plot_prob2_hin_sensitivity(sensitivity15=None,
                                   sensitivity10=None,
                                   args=[70, 75, 80],
                                   save=True, name="prob2"):
    if sensitivity15 is None:
        sensitivity15 = []
        sensitivity10 = []
        steps = [1600, 1800, 2500]
        for i, arg in enumerate(args):
            he = HeatEquation(steps[i], hin=arg/3.5)
            he.solve()
            sensitivity15.append(he.time_before(15))
            sensitivity10.append(he.time_before(10))
        print(sensitivity15)
        print(sensitivity10)
    plt.figure()
    plt.plot(args, sensitivity15, label="体表到达15度", marker='s')
    plt.plot(args, sensitivity10, label="体表到达10度", marker='s')
    plt.xlabel("新陈代谢速率(瓦/平方米)")
    plt.ylabel("时间(秒)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//hin_sensitivity_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


# plot_prob3([0.0006, 0.0009, 0.0012], [632.76, 727.18, 718.16],
#            [0.0010091346153846153, 0.000879326923076923, 0.0007495192307692307])
def plot_prob3(x, y1, y2, save=True, name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = np.array(x)*1000
    ax1.plot(x, y1, label="体表降到15度时间", marker='s')
    ax1.set_ylabel('体表降到15度时间(秒)')
    plt.xlabel('最外层厚度(毫米)')
    ax1.legend(loc=1)
    # ax1.set_yticks([3, 4, 5])
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, c="orange", label="织物层厚度", marker='s')
    ax2.set_ylabel('织物层厚度(毫米)')
    # ax2.set_ylim([0, 17])
    ax2.legend(loc=2)
    plt.tight_layout()

    if save:
        pdf = PdfPages("..//img//time_prob3{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def plot_prob3_3d(x1, t, x2, save=False, name=""):

    x1 = np.array(x1)*1000
    x2 = np.array(x2)*1000
    t = np.array(t)
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax = Axes3D(fig)

    x = [x1[0], x1[0], x1[1],  x1[2],  x1[2]]
    y = [x2[0], x2[0], x2[1],  x2[2],  x2[2]]
    z = [    0,  t[0],  t[1],   t[2],      0]

    # x = [0, 0.2, 1, 1, ]
    # y = [0, 0.2, 1, 1, ]
    # z = [0, 1, .7, 0, ]
    verts = [list(zip(x, y, z))]

    ax.add_collection3d(Poly3DCollection(verts, facecolors=['y'], alpha=.2))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x1[0]-0.1, x1[2]+0.1)
    ax.set_ylim(x2[2]-0.1, x2[0]+0.1)
    ax.set_zlim(0, t[1])
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//time_prob3{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_prob4_sensitivity(sensitivity15=None,
                           dsc_scales=np.arange(7, 8.6, 0.25),
                           hins=[70, 75, 80],
                           houts=[2.9, 3, 3.1],
                           save=True, name="prob4"):
    plt.figure()
    steps = [900, 800, 800]
    plt.plot([dsc_scales[0], dsc_scales[-1]], [727, 727], label="727秒", c="black", dashes=(5, 5))
    plt.scatter([7.911], [727], c="r", label="达到问题3坚持时间")
    for i, hout in enumerate(houts):
        sensitivity15 = []
        for dsc_scale in dsc_scales:
            he = HeatEquation(steps[i], hout=hout*2, dsc_scale=dsc_scale)
            he.solve()
            sensitivity15.append(he.time_before(15))
        print("hout={}, sensitivity15={}".format(hout, sensitivity15))
        plt.plot(dsc_scales, sensitivity15, label="外层对流换热系数{}".format(hout), marker='s')

    plt.xlabel("放热能力提高倍数")
    plt.ylabel("体表到达15度时间(秒)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//dsc_hout_sensitivity_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()
    #
    plt.figure()
    plt.plot([dsc_scales[0], dsc_scales[-1]], [727, 727], label="727秒", c="black", dashes=(5, 5))
    plt.scatter([7.911], [727], c="r", label="达到问题3坚持时间")
    for i, hin in enumerate(hins):
        sensitivity15 = []
        for dsc_scale in dsc_scales:
            he = HeatEquation(800, hin=hin/3.5, dsc_scale=dsc_scale)
            he.solve()
            sensitivity15.append(he.time_before(15))
        print("hin={}, sensitivity15={}".format(hin, sensitivity15))
        plt.plot(dsc_scales, sensitivity15, label="新陈代谢速率{}(瓦/平方米)".format(hin), marker='s')

    plt.xlabel("放热能力提高倍数")
    plt.ylabel("体表到达15度时间(秒)")
    plt.legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("..//img//dsc_hin_sensitivity_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

