import numpy as np
import pandas as pd
from tqdm import tqdm
show_progress = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
import read_data
import plot

# 建立“蚂蚁”类
class Ant(object):
    def __init__(self, path):
        self.path = path                       # 蚂蚁当前迭代整体路径
        self.unload_time = 1/12                #下货时间
        self.speed = 50                        #平均速度 km/h

    def permutation_path(self):
        # path=[A, B, C, D, A]注意路径闭环，将路径重新排列成配送站为出发点（A）
        permuted = np.zeros_like(self.path)
        warehouseID = np.where(self.path[:, 2] == 0)[0][0]
        pathSize = self.path.shape[0]
        permuted[:(pathSize-warehouseID)] = self.path[warehouseID:]
        permuted[(pathSize-warehouseID):]= self.path[1:warehouseID+1]
        self.path = permuted

    def calc_length(self):
        # 计算路径的总长度
        length = 0
        for i in range(len(self.path)-1):
            delta = np.abs(self.path[i, 0] - self.path[i+1, 0]) + np.abs(self.path[i, 1] - self.path[i+1, 1])
            length += delta
        return length

    def calc_time(self):
        # 计算路径的配送时间
        return self.calc_length()/self.speed + self.path.shape[0] * self.unload_time

    def calc_cost(self, has_small_cart=True):
        # 计算路径的总费用
        if self.path[0, 2] != 0:
            self.permutation_path()
        self.loaded_cost = 2
        self.total_demand = np.sum(self.path[:, 2])
        if self.total_demand <= 4 and has_small_cart:
            self.no_load_cost = 0.2
        elif self.total_demand <= 6:
            self.no_load_cost = 0.4
        else:
            self.no_load_cost = 0.6

        cost0, cost1 = 0, 0
        load = self.total_demand
        for i in range(len(self.path)-1):
            delta = np.abs(self.path[i, 0] - self.path[i+1, 0]) + np.abs(self.path[i, 1] - self.path[i+1, 1])
            if load > 1e-5:
                delta_cost = load * self.loaded_cost * delta
            else:
                delta_cost = self.no_load_cost * delta
            load -= self.path[i+1, 2]
            cost0 += delta_cost
        load = self.total_demand
        self.path = self.path[::-1, :]# 倒序排列
        for i in range(len(self.path)-1):
            delta = np.abs(self.path[i, 0] - self.path[i+1, 0]) + np.abs(self.path[i, 1] - self.path[i+1, 1])
            if load > 1e-5:
                delta_cost = load*self.loaded_cost * delta
            else:
                delta_cost = self.no_load_cost *delta
            load -= self.path[i+1, 2]
            cost1 += delta_cost
        return min(cost0, cost1)


# 建立“路径”类
class Path(object):
    def __init__(self, A, cities_num):                     # A为起始城市
        self.path = np.zeros((cities_num+1, 3))
        self.path[0] = A
        self.path[-1] = A
        self.step = 1

    def add_path(self, B):                     # 追加路径信息，方便计算整体路径长度
        self.path[self.step] = B
        self.step += 1


# 构建“蚁群算法”的主体
class ACO(object):
    def __init__(self, ant_num=40, maxIter=100, alpha=1, beta=5, rho=0.1, Q=1, df=None):
        self.ants_num = ant_num   # 蚂蚁个数
        self.maxIter = maxIter    # 蚁群最大迭代次数
        self.alpha = alpha        # 信息启发式因子
        self.beta = beta          # 期望启发式因子
        self.rho = rho            # 信息素挥发速度
        self.Q = Q                # 信息素强度
        #  提取所有城市的坐标信息
        if df is None:
            self.df = read_data.read_data()
        else:
            self.df = df
        self.deal_df(self.df)
        ###########################
        self.path_seed = np.zeros(self.ants_num).astype(int)      # 记录一次迭代过程中每个蚂蚁的初始城市下标
        self.ants_info = np.zeros((self.maxIter, self.ants_num))  # 记录每次迭代后所有蚂蚁的路径长度信息
        # 记录每次迭代后整个蚁群的“历史”最短路径长度，和该路径的总运费
        self.best_length = np.zeros(self.maxIter)
        self.best_cost = 0
        ###########################
        # self.solve()              # 完成算法的迭代更新
        # self.display()            # 数据可视化展示

    def deal_df(self, df):
        self.cities_num = len(df)                                                   # 1. 获取城市个数
        #  2. 构建城市列表
        self.cities = np.array(df[["X", "Y", "requirement"]])
        self.city_dist_mat = np.zeros((self.cities_num, self.cities_num))                  # 3. 构建城市距离矩阵
        for i in range(self.cities_num):
            self.city_dist_mat[i] = np.abs(self.cities[i, 0] - self.cities[:, 0]) \
                                    + np.abs(self.cities[i, 1] - self.cities[:, 1])
        plot.plot_mat(self.city_dist_mat, name="dist_mat_TS")
        self.phero_mat = np.ones((self.cities_num, self.cities_num))                       # 4. 初始化信息素矩阵
        # self.phero_upper_bound = self.phero_mat.max() * 1.2                              #b信息素浓度上限
        self.eta_mat = 1/(self.city_dist_mat + np.diag([np.inf]*self.cities_num))          # 5. 初始化启发函数矩阵

    def solve(self, save_img_name=None, verbose=False, has_small_cart=True):
        for iterNum in tqdm(range(self.maxIter)) if show_progress else range(self.maxIter):
            self.random_seed()                                                 # 使整个蚁群产生随机的起始点
            delta_phero_mat = np.zeros((self.cities_num, self.cities_num))     # 初始化每次迭代后信息素矩阵的增量
            for i in range(self.ants_num):
                city_index1 = self.path_seed[i]                                # 每只蚂蚁访问的第一个城市下标
                ant_path = Path(self.cities[city_index1], self.cities_num)     # 记录每只蚂蚁访问过的城市
                tabu = [city_index1]                                           # 记录每只蚂蚁访问过的城市下标，禁忌城市下标列表
                non_tabu = list(set(range(self.cities_num)) - set(tabu))
                for j in range(self.cities_num-1):                             # 对余下的城市进行访问
                    up_proba = np.zeros(self.cities_num-len(tabu))             # 初始化状态迁移概率的分子
                    up_proba = np.power(self.phero_mat[city_index1][non_tabu], self.alpha) * \
                        np.power(self.eta_mat[city_index1][non_tabu], self.beta)
                    proba = up_proba/sum(up_proba)                             # 每条可能子路径上的状态迁移概率
                    proba_cumsum = np.cumsum(proba)
                    # 提取出下一个城市的下标
                    choicesArr = np.where(proba_cumsum > np.random.rand())[0]
                    if choicesArr.size == 0:
                        choice = -1
                    else:
                        choice = choicesArr[0]
                    city_index2 = non_tabu[choice]

                    ant_path.add_path(self.cities[city_index2])
                    tabu.append(city_index2)
                    non_tabu = list(set(range(self.cities_num)) - set(tabu))
                    city_index1 = city_index2
                self.ants_info[iterNum][i] = Ant(ant_path.path).calc_length()
                if iterNum == 0 and i == 0:                                    # 完成对最佳路径城市的记录
                    self.best_cities = ant_path.path
                else:
                    if self.ants_info[iterNum][i] < Ant(self.best_cities).calc_length():
                        self.best_cities = ant_path.path
                tabu.append(tabu[0])                                           # 每次迭代完成后，使禁忌城市下标列表形成完整闭环
                for l in range(self.cities_num):
                    delta_phero_mat[tabu[l]][tabu[l+1]] += self.Q/self.ants_info[iterNum][i]

            self.best_length[iterNum] = Ant(self.best_cities).calc_length()
            self.update_phero_mat(delta_phero_mat)                             # 更新信息素矩阵
        best_ant = Ant(self.best_cities)
        self.best_cost = best_ant.calc_cost(has_small_cart)
        self.total_time = best_ant.calc_time()
        if save_img_name is not None:
            plot.display_TS(self.df, save_img_name, self.ants_info, self.best_length, self.best_cities)
        if verbose:
            print("\nbest result: {} {}.\n".format(self.best_length[-1], self.best_cost))
        return self.best_length[-1], self.best_cost, self.total_time

    def update_phero_mat(self, delta):
        self.phero_mat = (1 - self.rho) * self.phero_mat + delta

    def random_seed(self):                                                     # 产生随机的起始点下表，尽量保证所有蚂蚁的起始点不同
        if self.ants_num <= self.cities_num:                                   # 蚂蚁数 <= 城市数
            self.path_seed[:] = np.random.permutation(range(self.cities_num))[:self.ants_num]
        else:                                                                  # 蚂蚁数 > 城市数
            self.path_seed[:self.cities_num] = np.random.permutation(range(self.cities_num))
            temp_index = self.cities_num
            while temp_index + self.cities_num <= self.ants_num:
                self.path_seed[temp_index:temp_index + self.cities_num] = np.random.permutation(range(self.cities_num))
                temp_index += self.cities_num
            temp_left = self.ants_num % self.cities_num
            if temp_left != 0:
                self.path_seed[temp_index:] = np.random.permutation(range(self.cities_num))[:temp_left]


class ACO_mutiTS():
    def __init__(self, ts_num, df=None):
        self.ts_num = ts_num
        self.total_distance = 0
        self.total_cost = 0
        self.time_consumed = 0
        if df is None:
            self.df = read_data.read_data()
        else:
            self.df = df
        self.cities = self.df.loc[(self.df["X"] != 10) | (self.df["Y"] != 10)]
        self.warehouse = self.df.loc[(self.df["X"] == 10) & (self.df["Y"] == 10)]
        self.warehouse["cluster"] = -1
        self.ACO_model_ls = []
        # self.init_cluster_kmeans()

    def init_cluster_kmeans(self, save_img_name=None):
        load_type = []
        clustering_model = KMeans(n_clusters=self.ts_num, init='k-means++', n_jobs=-1)

        self.cities["cluster"] = clustering_model.fit_predict(self.cities)
        for cls in np.unique(self.cities["cluster"]):
            df_cluster = self.cities.loc[self.cities["cluster"] == cls]
            cluster_demand = df_cluster["requirement"].sum()
            if cluster_demand <= 4:
                load_type.append(0)
            elif cluster_demand <= 6:
                load_type.append(1)
            else:
                load_type.append(2)
            df_cluster = df_cluster.append(self.warehouse, ignore_index=True)
            self.ACO_model_ls.append(ACO(df=df_cluster))
        if save_img_name:
            plot.plot_cluster_map(self.cities, name=save_img_name)
        return load_type


    def init_cluster_custom(self, clusterdf):
        self.cities = clusterdf
        print("\n requirement of clusters: ")
        for cls in np.unique(self.cities["cluster"]):
            df_cluster = self.cities.loc[self.cities["cluster"] == cls]
            print("{}".format(df_cluster["requirement"].sum()),end=" ")
            df_cluster = df_cluster.append(self.warehouse, ignore_index=True)
            self.ACO_model_ls.append(ACO(df=df_cluster))
        print("\n")

    def solve(self, save_img_name=None, verbose=True, has_small_cart=True):
        for ACO_model in self.ACO_model_ls:
            best_distance, best_cost, time_consumed= ACO_model.solve(has_small_cart=has_small_cart)
            self.total_distance += best_distance
            self.total_cost += best_cost
            if time_consumed > self.time_consumed:
                self.time_consumed = time_consumed

        if save_img_name:
            plot.display_mutiTS(self.cities, save_img_name, self.ACO_model_ls)
        if verbose:
            print("\nFinal best result is {} km.\n".format(self.total_distance))
        return self.total_distance, self.total_cost, self.time_consumed


def test_ACO():
    model_single = ACO()
    model_single.solve(save_img_name="TS", verbose=True)


def test_ACO_mutiTS_kmeans6(pointSize=20):
    while 1:
        plt.figure(figsize=(12, 18))
        ts_num_ls = [4, 5, 6, 7, 8, 9]
        best_distance_ls = []
        best_cost_ls = []
        load_type_ls = []
        time_consumed_ls = []
        for i, ts_num in enumerate(ts_num_ls):
            model = ACO_mutiTS(ts_num)
            load_type = model.init_cluster_kmeans()
            load_type_ls.append(max(load_type))
            best_distance, best_cost, time_consumed = model.solve(has_small_cart=False)
            best_distance_ls.append(best_distance)
            best_cost_ls.append(best_cost)
            time_consumed_ls.append(time_consumed)
            plt.subplot(3, 2, i+1)
            plt.scatter(model.cities["X"], model.cities["Y"], s=pointSize*model.cities["requirement"], c=model.cities["cluster"])
            plt.scatter([10.], [10.], s=[pointSize], c="r")
            for cluster, ACO_model in enumerate(model.ACO_model_ls):
                plt.plot(ACO_model.best_cities[:, 0], ACO_model.best_cities[:, 1], dashes=[6, 2])
            plt.title("time:{:.1f}, cost:{:.2f}".format(time_consumed, best_cost))
            if i==8 or i==9:
                print(model.cities)
        if load_type_ls[-2]<2 and load_type_ls[-1]<2:
            break
    pdf = PdfPages("img//path_mutiTS_6pic.pdf")
    pdf.savefig()
    pdf.close()
    plt.close()

    xlabel = 'number of carrying cart'
    # 画距离成本图
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(ts_num_ls, best_distance_ls, "b-")
    ax1.set_ylabel('total distance (km)', color='b')
    ax1.set_xlabel(xlabel)
    ax2.plot(ts_num_ls, best_cost_ls, "g-")
    ax2.set_ylabel('total cost (yuan)', color='g')
    for i, load_type in enumerate(load_type_ls):
        if load_type == 2:
            ax1.scatter([ts_num_ls[i]], [best_distance_ls[i]], 2 * pointSize, "r", marker="x")
            ax2.scatter([ts_num_ls[i]], [best_cost_ls[i]], 2 * pointSize, "r", marker="x")

    pdf = PdfPages("img//distance_cost_line.pdf")
    pdf.savefig()
    pdf.close()
    plt.close()

    # 画时间成本图
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(ts_num_ls, time_consumed_ls, "b-")
    ax1.set_ylabel('total time consumed (h)', color='b')
    ax1.set_xlabel(xlabel)
    ax2.plot(ts_num_ls, best_cost_ls, "g-")
    ax2.set_ylabel('total cost (yuan)', color='g')
    for i, load_type in enumerate(load_type_ls):
        if load_type == 2:
            ax1.scatter([ts_num_ls[i]], [time_consumed_ls[i]], 2*pointSize, "r", marker="x")
            ax2.scatter([ts_num_ls[i]], [best_cost_ls[i]], 2* pointSize, "r", marker="x")

    pdf = PdfPages("img//time_cost_line.pdf")
    pdf.savefig()
    pdf.close()
    plt.close()
    print(load_type_ls)
    print(best_cost_ls)
    print(time_consumed_ls)
    print(best_distance_ls)


def test_ACO_mutiTS_kmeans3(pointSize=20):
    plt.figure(figsize=(6, 18))
    ts_num_ls = [7, 8, 9]
    best_distance_ls = []
    best_cost_ls = []
    load_type_ls = []
    time_consumed_ls = []
    for i, ts_num in enumerate(ts_num_ls):
        model = ACO_mutiTS(ts_num)
        load_type = model.init_cluster_kmeans()
        print(load_type)
        load_type_ls.append(max(load_type))
        best_distance, best_cost, time_consumed = model.solve()
        best_distance_ls.append(best_distance)
        best_cost_ls.append(best_cost)
        time_consumed_ls.append(time_consumed)
        plt.subplot(3, 1, i+1)
        plt.scatter(model.cities["X"], model.cities["Y"], s=pointSize*model.cities["requirement"], c=model.cities["cluster"])
        plt.scatter([10.], [10.], s=[pointSize], c="r")
        for cluster, ACO_model in enumerate(model.ACO_model_ls):
            plt.plot(ACO_model.best_cities[:, 0], ACO_model.best_cities[:, 1], dashes=[6, 2])
        plt.title("total distance:{:.1f}, cost:{:.2f}".format(best_distance, best_cost))
    pdf = PdfPages("img//path_mutiTS_3pic.pdf")
    pdf.savefig()
    pdf.close()
    plt.close()

def test_ACO_mutiTS_custom():
    for sheet_name in ("Sheet2",):
        clusterdf = pd.read_excel("data//cluster//cluster.xls", sheet_name=sheet_name)
        ts_num = np.unique(clusterdf["cluster"]).size
        print(sheet_name + " ts_num: {}".format(ts_num))
        model = ACO_mutiTS(ts_num)
        model.init_cluster_custom(clusterdf)
        best_distance, best_cost, time_consumed = model.solve(save_img_name=sheet_name)
        print(best_distance, best_cost, time_consumed)


if __name__ == '__main__':
# test_ACO()
    #test_ACO_mutiTS_custom()
    test_ACO_mutiTS_kmeans6()


