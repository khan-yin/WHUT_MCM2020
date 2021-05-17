from common import *
from heat_pde import *


def plot_prob1():
    he = HeatEquation(1600)
    name = "prob1"
    he.solve()
    print("皮肤表面降温到{}摄氏度的时间为{:.2f}秒".format(15, he.time_before(15)))
    print("皮肤表面降温到{}摄氏度的时间为{:.2f}秒".format(10, he.time_before(10)))
    # 皮肤表面降温到15摄氏度的时间为557.04秒
    # 皮肤表面降温到10摄氏度的时间为1524.96秒
    he.plot_T_slice([he.time_before(15), he.time_before(10)], name=name)
    he.plot_x_slice([he.time_before(15), he.time_before(10)], name=name)
    he.plot_T_surface(name=name)


def plot_prob2():
    he = HeatEquation(1000, hout=8, hin=80/3.5)
    name = "prob2"
    he.solve()
    print("皮肤表面降温到{}摄氏度的时间为{:.2f}秒".format(15, he.time_before(15)))
    print("皮肤表面降温到{}摄氏度的时间为{:.2f}秒".format(10, he.time_before(10)))
    he.plot_T_slice([he.time_before(15), he.time_before(10)], name=name)
    he.plot_x_slice([he.time_before(15), he.time_before(10)], name=name)
    he.plot_T_surface(name=name)


def plot_prob3():
    he = HeatEquation()
    he.get_delta_tk()


def plot_prob4():
    dsc_scale = 7.911
    he = HeatEquation(1700, dsc_scale=dsc_scale)
    name = "prob4"
    he.solve()
    result = he.time_before(15)
    print("dsc_scale={}, result={}".format(dsc_scale, result))
    he.plot_T_slice([he.time_before(15), he.time_before(10)], name=name)
    he.plot_x_slice([he.time_before(15), he.time_before(10)], name=name)


if __name__ == '__main__':
    plot_prob1()
    plot_prob2()
    plot_prob3()
    plot_prob4()
