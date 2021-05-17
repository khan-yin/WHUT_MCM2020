import numpy as np
import data
import plot
import functools
import time
from tqdm import tqdm
import collections
import utils
INF = np.inf
MAX_LOAD = 40
TRY_MAX_AVAILABLE_I = 5
TRY_MAX_PERCENTAGE = 0.9
DATA = data.Data()


# 计时装饰器
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - begin_time
        print('{} 共用时：{} s'.format(func.__name__, run_time))
        return value
    return wrapper


class Tree:
    def __init__(self, edges):
        self.edges = set(edges)
        self.tree, self.front_station_of = self.edges_to_tree(edges)
        self.cal_length_from_all_station()
        self.IFirstID = DATA.stationNum[0]
        self.ILastID = DATA.stationNum[0] + DATA.stationNum[1]

    @staticmethod
    def distece_between(v, u):
        return DATA.distanceMat12[v, u]
    @staticmethod
    def edges_to_tree(edges):
        station = DATA.station012
        tree = dict()
        front_station_of = dict()
        for stationID in station.loc[station["classID"] != 0]["id"]:
            tree[stationID] = set()

        for k, v in edges:
            if k in tree:
                tree[k].add(v)
                front_station_of[v] = k
        return tree, front_station_of

    def cal_length_from_all_station(self):
        self.length_from_station = dict()
        station = DATA.station012
        IStationIDLst = station.loc[station["classID"] == 1]["id"]
        for IStationID in IStationIDLst:
            self.cal_length_from_station(IStationID)

    def cal_length_from_station(self, stationID):
        """
        递归求解从当前stationID输出的所有供水站管线长度和
        :param stationID:
        :return:
        """
        if len(self.tree[stationID]) == 0:
            self.length_from_station[stationID] = 0
            return 0
        elif stationID in self.length_from_station:
            return self.length_from_station[stationID]
        else:
            length = 0
            for next in self.tree[stationID]:
                length += DATA.distanceMat12[stationID, next]
                length += self.cal_length_from_station(next)
            self.length_from_station[stationID] = length
            return length

    def get_availableIStation_overloadIStationSet(self):
        availableIStation2availableLoad = dict()
        overloadIStationSet = set()
        for i in range(self.IFirstID, self.ILastID):
            load = self.length_from_station[i]
            if load < TRY_MAX_PERCENTAGE*MAX_LOAD:
                availableIStation2availableLoad[i] = MAX_LOAD - load
            elif load > MAX_LOAD:
                overloadIStationSet.add(i)
        return availableIStation2availableLoad, overloadIStationSet

    def get_grafting_candidate(self):
        availableI2Load, overloadIStationSet = self.get_availableIStation_overloadIStationSet()
        if len(availableI2Load) == 0:
            return None, None
        # 对availableLoad进行降序排列
        self.orderedAvailableI2Load = collections.OrderedDict(
            sorted(availableI2Load.items(), key=lambda t: -t[1])
        )

        self.orderedAvailableI2Station = collections.OrderedDict()
        for i, availableI in enumerate(self.orderedAvailableI2Load.keys()):
            self.orderedAvailableI2Station[availableI] = set((availableI,))
            self.add_all_next_selectedAvailable(availableI, availableI)
            if i >= TRY_MAX_AVAILABLE_I:
                break

        self.orderedAvailableI2candidateOverload = collections.OrderedDict()
        for i, availableI in enumerate(self.orderedAvailableI2Load.keys()):
            self.orderedAvailableI2candidateOverload[availableI] = set()
            for overloadIstation in overloadIStationSet:
                self.add_all_next_candidateOverload(availableI, overloadIstation)
            if i >= TRY_MAX_AVAILABLE_I:
                break
        return self.orderedAvailableI2Station, self.orderedAvailableI2candidateOverload

    def add_all_next_selectedAvailable(self, availableI, stationID):
        """
        递归地把树中所有stationID的子节点加入self.orderedAvailableI2Station[]
        :param stationID:
        :return:
        """
        if len(self.tree[stationID]) == 0:
            # 无下一节点
            return
        for next in self.tree[stationID]:
            self.orderedAvailableI2Station[availableI].add(next)
            self.add_all_next_selectedAvailable(availableI, next)

    def add_all_next_candidateOverload(self, availableI, stationID):
        if len(self.tree[stationID]) == 0:
            # 无下一节点
            return
        for next in self.tree[stationID]:
            load = self.length_from_station[next]
            if load < self.orderedAvailableI2Load[availableI]:
                self.orderedAvailableI2candidateOverload[availableI].add(next)
            self.add_all_next_candidateOverload(availableI, next)

    def get_cutEdge(self, graftEdge, availableI):
        u, v = graftEdge
        graftEdgeLength = self.distece_between(u, v)
        if graftEdgeLength > self.orderedAvailableI2Load[availableI]:
            return None
        self.edges.add((u, v))
        stationID = v
        while self.length_from_station[self.front_station_of[stationID]] < \
                self.orderedAvailableI2Load[availableI]-graftEdgeLength:
            self.edges.discard((self.front_station_of[stationID], stationID))
            self.edges.add((stationID, self.front_station_of[stationID]))
            stationID = self.front_station_of[stationID]
        cutEdge = (self.front_station_of[stationID], stationID)
        self.edges.discard(cutEdge)
        self.tree, self.front_station_of = self.edges_to_tree(self.edges)
        self.cal_length_from_all_station()
        return cutEdge

class Prim:
    def __init__(self):
        self.pip1 = []
        self.pip1len = 0
        self.pip2 = []
        self.pip2len = 0

        self.solve1()
        self.solve2()
        self.solve3()
    @staticmethod
    def min_edge(selected, candidate, graph, idMap=None):
        """
        求已经确定的顶点集合与未选顶点集合中的最小边
        :param selected:
        :param candidate:
        :param graph:
        :param idMap: 将graph中的索引映射到station["ID"]
        :return:返回新增选择的编号以及记录的最小边的两个顶点的station["ID"]
        """
        v, u = 0, 0
        weights = np.zeros((len(selected), len(candidate)))
        selectedArr = np.array(list(selected))
        candidateArr = np.array(list(candidate))
        # 循环扫描已选顶点与未选顶点，寻找最小边
        for i in range(len(selected)):
            weights[i] = graph[selectedArr[i], candidateArr]
        minID = np.argmin(weights)
        i = selectedArr[minID // len(candidate)]
        j = candidateArr[minID % len(candidate)]

        if idMap is not None:
            v, u = idMap[i], idMap[j]
        else:
            v, u = i, j
        return j, v, u

    def prim(self, graph, selected=None, candidate=None, idMap=None):
        """"""
        if selected is None:
            selected = set((0,))
        if candidate is None:
            candidate = set(range(len(graph))) - selected
        # 顶点个数
        vertex_num = len(candidate)
        # 存储每次搜索到的最小生成树的边
        edges = []
        # 由于连接n个顶点需要n-1条边，故进行n-1次循环，以找到足够的边
        for i in range(vertex_num):
            # 调用函数寻找当前最小边
            selection, v, u = self.min_edge(selected, candidate, graph, idMap=idMap)
            # 添加到最小生成树边的集合中
            edges.append((v, u))
            # v是selected中的顶点，u为candidate中的顶点，故将u加入candidate，以代表已经选择该顶点
            selected.add(selection)
            # 同时将u从candidate中删除
            candidate.discard(selection)
        return edges

    def cal_length(self, edges):
        """
        求生成树的总长度
        :param edges:
        :return:生成树的总长度，平均长度
        """
        station = DATA.station012
        total_length = 0
        for edge in edges:
            x0, x1 = station["X"][edge[0]], station["X"][edge[1]]
            y0, y1 = station["Y"][edge[0]], station["Y"][edge[1]]
            total_length += np.sqrt((x0-x1)**2 + (y0-y1)**2)
        return total_length, total_length/len(edges)

    @timer
    def solve1(self):
        """
        使用prim算法求解第1问，I型II型管道的方案
        :return:
        """
        self.pip1 = self.prim(DATA.distanceMat01)
        print("I型管道{}条".format(len(self.pip1)))
        self.pip1len, self.pip1avglen = self.cal_length(self.pip1)
        print("total length: {}".format(self.pip1len))

        selectedNum = DATA.stationNum[0] + DATA.stationNum[1]
        print("selectedNum: {}".format(selectedNum))

        self.pip2 = self.prim(DATA.distanceMat12, selected=set(range(selectedNum)))
        print("II型管道{}条".format(len(self.pip2)))
        print(self.pip2)
        self.pip2len, self.pip2avglen = self.cal_length(self.pip2)
        print("total length: {}".format(self.pip2len))
        plot.plot_pipeline(DATA.station012, self.pip1, self.pip2)

        print("I型供水站总负载")
        self.tree2 = Tree(self.pip2)
        # print(self.tree2.length_from_station)
        station = DATA.station012
        IStationIDLst = station.loc[station["classID"] == 1]["id"]
        for IStationID in IStationIDLst:
            print("{}: {}, ".format(IStationID, self.tree2.length_from_station[IStationID]))

    @timer
    def solve2(self, skip=True):
        """
        使用枚举和prim算法求解第2问，搜索升级两个二级水站后，II型管道总长度最小的方案
        :return:
        """
        vertex_num = len(DATA.distanceMat12)# 顶点个数
        selectedNum = DATA.stationNum[0] + DATA.stationNum[1]
        selected = set(range(selectedNum))
        candidate = set(range(vertex_num)) - selected
        candidateLst = list(candidate)
        self.min_lengthAfterUpgrade = self.pip2len

        if skip:
            upgradeStationID = set((89, 125))
            self.upgradeStationID = upgradeStationID
            selectedAfterUpgrade = selected | upgradeStationID
            pip2 = self.prim(DATA.distanceMat12, selected=selectedAfterUpgrade)
            self.min_pip2 = pip2
            pip2len, _ = self.cal_length(pip2)
        else:
            trials = [(i, j) for i in range(1, len(candidateLst)) for j in range(i)]
            total_step = len(trials)
            for i, j in tqdm(trials):
                upgradeStationID = set((candidateLst[i], candidateLst[j]))
                selectedAfterUpgrade = selected | upgradeStationID
                pip2 = self.prim(DATA.distanceMat12, selected=selectedAfterUpgrade)
                pip2len, _ = self.cal_length(pip2)
                if pip2len < self.min_lengthAfterUpgrade:
                    self.min_lengthAfterUpgrade = pip2len
                    self.min_pip2 = pip2
                    self.upgradeStationID = upgradeStationID

        print("分别升级{}".format(self.upgradeStationID))
        print("\n升级两个二级水站后的II型管道{}条".format(len(self.min_pip2)))
        print(self.min_pip2)
        print("II型管道最小总长度: {}".format(self.min_lengthAfterUpgrade))
        upgradeStation = DATA.station012.loc[self.upgradeStationID]
        station01AfterUpgrade = DATA.station01.append(upgradeStation).reset_index(drop=True)
        graphAfterUpgrade = DATA.get_distanceMat(station01AfterUpgrade)
        self.min_pip1 = self.prim(graphAfterUpgrade, idMap=station01AfterUpgrade["id"])
        print("\n升级两个二级水站后的I型管道{}条".format(len(self.min_pip1)))
        print(self.min_pip1)
        plot.plot_pipeline(DATA.station012, self.min_pip1, self.min_pip2,
                           upgradeStationID=list(self.upgradeStationID), name="pipelineAfterUpgrade")

    @timer
    def solve3(self):
        i = 0
        lastCutEdge = (-1, -1)
        plot.plot_tree(DATA.station012, self.pip1, self.tree2, name="pipline_length")
        while True:
            orderedAvailableI2Station, orderedAvailableI2candidateOverload =\
                self.tree2.get_grafting_candidate()
            if orderedAvailableI2Station is None:
                break
            for availableI in orderedAvailableI2Station.keys():
                cutEdge = None
                selectedAvailable = orderedAvailableI2Station[availableI]
                candidateOverload = orderedAvailableI2candidateOverload[availableI]
                if len(selectedAvailable) == 0 or len(candidateOverload) == 0:
                    continue
                # 调用函数寻找当前最小边
                selection, v, u = self.min_edge(selectedAvailable, candidateOverload,
                                                DATA.distanceMat12)
                plot.plot_tree(DATA.station012, self.pip1, self.tree2,
                               selectedAvailable, candidateOverload,
                               extraEdges=((v, u),),annotateIIStation=False,
                               name="pipline_graft_connection_{}".format(i))
                cutEdge = self.tree2.get_cutEdge((v, u), availableI)
                if cutEdge is not None:
                    break
            if cutEdge == lastCutEdge:
                break
            lastCutEdge = cutEdge
            if cutEdge is not None:
                plot.plot_tree(DATA.station012, self.pip1, self.tree2,
                               extraEdges=((v, u), cutEdge), cutEdge=cutEdge, annotateIIStation=False,
                               name="pipline_graft_cut_{}".format(i))
            i += 1

        plot.plot_tree(DATA.station012, self.pip1, self.tree2,
                       cutEdge=cutEdge, annotateIIStation=False,
                       name="pipline_graft_cut_{}".format(i))
        print("嫁接后II级水管总长：{}".format(self.cal_length(self.tree2.edges)[0]))


if __name__ == '__main__':
    p = Prim()

    


