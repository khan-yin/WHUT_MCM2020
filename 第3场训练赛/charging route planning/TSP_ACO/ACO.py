import numpy as np
import pandas as pd
from tqdm import tqdm
show_progress = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from geopy.distance import geodesic, great_circle
from sklearn.cluster import KMeans
import TSP_ACO.read_data as read_data
import TSP_ACO.plot as plot

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
    def __init__(self, ant_num=40, maxIter=500, alpha=1, beta=3, rho=0.001, Q=1, best_praise=1, df=None):
        self.ants_num = ant_num   # 蚂蚁个数
        self.maxIter = maxIter    # 蚁群最大迭代次数
        self.alpha = alpha        # 信息启发式因子
        self.beta = beta          # 期望启发式因子
        self.rho = rho            # 信息素挥发速度
        self.Q = Q                # 信息素强度
        self.best_praise = best_praise
        #  提取所有城市的坐标信息
        if df is None:
            self.df = read_data.read_data()
        else:
            self.df = df
        self.deal_df()
        ###########################
        self.path_seed = np.zeros(self.ants_num).astype(int)      # 记录一次迭代过程中每个蚂蚁的初始城市下标
        self.ants_info = np.zeros((self.maxIter, self.ants_num))  # 记录每次迭代后所有蚂蚁的路径长度信息
        # 记录每次迭代后整个蚁群的“历史”最短路径长度，和该路径的总运费
        self.best_length = np.zeros(self.maxIter)
        # self.best_cost = 0
        ###########################
        # self.solve()              # 完成算法的迭代更新
        # self.display()            # 数据可视化展示

    def deal_df(self):
        self.cities_num = len(self.df)                                                   # 1. 获取城市个数
        #  2. 构建城市列表
        self.cities = np.array(self.df[["longitude", "latitude", "TDP"]])
        # print(self.df)
        try:
            self.city_dist_mat = np.array(pd.read_excel("data/dist_mat.xlsx"))[:, 1:]
        except FileNotFoundError:
            self.city_dist_mat = np.zeros((self.cities_num, self.cities_num))                  # 3. 构建城市距离矩阵
            for i in range(self.cities_num):
                for j in range(self.cities_num):
                    cityi = (self.df["latitude"][i], self.df["longitude"][i])
                    cityj = (self.df["latitude"][j], self.df["longitude"][j])
                    self.city_dist_mat[i][j] = great_circle(cityi, cityj).km
            pd.DataFrame(self.city_dist_mat).to_excel("data/dist_mat.xlsx")
        plot.plot_mat(self.city_dist_mat, name="dist_mat_TS")
        self.phero_mat = np.ones((self.cities_num, self.cities_num))                       # 4. 初始化信息素矩阵
        # self.phero_upper_bound = self.phero_mat.max() * 1.2                              #b信息素浓度上限
        self.eta_mat = 1/(self.city_dist_mat + np.diag([np.inf]*self.cities_num))          # 5. 初始化启发函数矩阵

    def calc_length(self, tabu):
        length = 0
        for i in range(len(tabu)-1):
            length += self.city_dist_mat[tabu[i], tabu[i+1]]
        return length

    def solve(self, save_img_name=None, verbose=False):
        for iterNum in tqdm(range(self.maxIter)) if show_progress else range(self.maxIter):
            #self.random_seed()                                                # 使整个蚁群产生随机的起始点
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
                tabu.append(tabu[0])                                           # 每次迭代完成后，使禁忌城市下标列表形成完整闭环
                is_best = False
                self.ants_info[iterNum][i] = self.calc_length(tabu)
                if iterNum == 0 and i == 0:                                    # 完成对最佳路径城市的记录
                    self.best_cities = ant_path.path
                    self.best_cities_id = tabu
                else:
                    if self.ants_info[iterNum][i] < self.calc_length(self.best_cities_id):
                        is_best = True
                        self.best_cities = ant_path.path
                        self.best_cities_id = tabu
                for l in range(self.cities_num):
                    delta_phero_mat[tabu[l]][tabu[l+1]] += self.Q*(self.best_praise**is_best)/self.ants_info[iterNum][i]


            self.best_length[iterNum] = self.calc_length(self.best_cities_id)
            self.update_phero_mat(delta_phero_mat)                             # 更新信息素矩阵

        if save_img_name is not None:
            plot.display_TS(self.df, save_img_name, self.ants_info, self.best_length, self.best_cities)
        if verbose:
            print("\nbest length: {}.\n".format(self.best_length[-1]))
        return self.best_length[-1]

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

def testDistance():
    newport_ri = (36.3742269854581, 120.701520187911)
    cleveland_oh = (36.3745756929757, 120.6987175)
    print(great_circle(newport_ri, cleveland_oh).km)
    print(geodesic(newport_ri, cleveland_oh).km)
    print()

if __name__ == '__main__':
    # testDistance()
    m = ACO()
    m.solve(save_img_name="problem1", verbose=True)
    print(m.best_cities_id)


