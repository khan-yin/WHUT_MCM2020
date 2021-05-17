from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import read_data
from solve_scipy import *
import plot
from common import *


#
# def get_seir_solution(p):
#     u0 = (p["S0"], 109, 20, 33, 5, 0, 0, 0)
#     parameters = (p["beta"], p["start"], p["hStart"], p["k"], p["epsilon"], p["lamb"],
#                   p["theta"], p["delta"], p["sigma"], p["gamma"], p["eta"])
#     t = np.arange(0, DAYS, 1)
#     seir_solution = odeint(f1, u0, t, args=parameters)
#     return seir_solution[:DAYS, -3:]
#
#
# def journey_objective(p):
#     """
#     模型超参数搜索优化总距离（成本）
#     :param p:
#     :return:
#     """
#     seir_solution = get_seir_solution(p)
#     # loss = (np.abs(DATA_JRD - seir_solution)).sum()
#     loss = np.abs(np.diff(seir_solution, axis=0)-DATA_JRD_DIFF).mean()
#     result = {"total_loss": loss}
#     return {'loss': result["total_loss"], "result": result, 'parameters': p, 'status': STATUS_OK}
#
#
# """
# best parameter found after 1000 trials
# {'S0': 11270.118222399033, 'beta': 4.2321373114743295e-05, 'delta': 0, 'epsilon': 0.1998471240213448, 'eta': 0, 'gamma': 0.07658716283023054, 'hStart': 13, 'k': 0.5107896354918074, 'lamb': 0.17242616434988262, 'sigma': 0.1919341966337201, 'start': 2, 'theta': 0.30693829040250115}
# lowest loss is
# {'total_loss': 1.8681512592460094}
# """
#
# def opt_journey(max_evals):
#     """
#     调用hyperopt库的API接口在搜索空间中确定最佳超参数优化总距离（成本）
#     :param max_evals:
#     :return:
#     """
#     # p = (1e-5, 0.1, 2/35, 3/35, 1/3, 1/3, 1/17, 0.1, 0.5)
#     space = {
#         "S0": hp.loguniform("S0", np.log(1e4), np.log(1e5)),
#         "beta": hp.loguniform("beta", np.log(1e-5), np.log(1e-4)),
#         "start": 2,#hp.choice("start", range(20)),#疫情开始传播
#         "hStart": hp.choice("hStart", range(10, 16)),#开始实行隔离政策的天数
#         "k": hp.uniform("k", 0.2, 0.9),#潜伏期患者与已发病患者传染强度的比率
#         "epsilon": hp.uniform("epsilon", 0, 1/5),#潜伏期感染者转入未隔离感染者的比例
#         "lamb": hp.uniform("lamb", 0, 1/2),#潜伏期感染者转入已隔离感染者的比例
#
#         "theta": hp.uniform("theta", 0.1, 0.4),#未隔离感人者转入正在治疗的人群的比例
#         "sigma": hp.uniform("sigma", 0.15, 0.21),#已隔离感人者转入正在治疗的人群的比例
#         "gamma": hp.uniform("gamma", 0.07, 0.08),#正在治疗的已感染者治愈的比例
#         "delta":0,# hp.uniform("delta", 0.05, 0.2),#未隔离的已感染者的死亡率
#         "eta":0,# hp.uniform("eta", 0, 0.05),#接受治疗的病人的死亡率
#     }
#     trials = Trials()
#     best = fmin(fn=journey_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
#     return best, trials
#
#
# def get_best_parameters_for_journey(max_evals):
#     """
#     封装opt_journey解析参数搜索结果
#     :param max_evals:
#     :return:
#     """
#     best, trials = opt_journey(max_evals)
#     # Sort the trials with lowest loss first
#     trials_list = sorted(trials.results, key=lambda x: x['loss'])
#     bestParameters = trials_list[0]['parameters']
#     result = trials_list[0]["result"]
#     # print(trials_list[0])
#     print("best parameter found after {} trials".format(len(trials_list)))
#     print(bestParameters)
#     print("lowest loss is")
#     print(result)
#     seir_solution = get_seir_solution(bestParameters)
#     plot.compare_result_with_data(seir_solution, DATA_JRD)
#     # plot.plot_trials(trials_list)
#


def get_medical_upgrade_solution(p):
    # u0 = (p["S0"], 109, 10, 33, 5, 0, 0, 0)
    u0 = (p["S0"], 109, p["I0"], p["Q0"], 0, 0, 0, 0)
    parameters = (p["betaTimesS0"] / p["S0"], p["start"], p["hStart"], p["k"], p["epsilon"], p["lamb"],
                  p["theta"], p["delta"], p["sigma"], p["gamma0"], p["gamma1"],
                  p["mStart"], p["mEnd"], p["eta"])
    t = np.arange(0, DAYS, 1)
    seir_solution = odeint(f2, u0, t, args=parameters)
    return seir_solution[:DAYS, -3:]


class MedicalUpgrade():
    def __init__(self, province, max_evals, start=None, hStart=None, mStart=None, mEnd=None):
        self.DATA_JRD = np.array(read_data.read_cleaned_data(province)[
                                ["confirmedMA", "recoveriesMA", "deathsMA"]][:DAYS].fillna(0))

        self.DATA_JRD_DIFF = np.diff(self.DATA_JRD, axis=0)
        self.max_evals = max_evals
        self.province = province

        self.space = {
            "S0": hp.loguniform("S0", np.log(1e4), np.log(1e5)),
            "betaTimesS0": hp.uniform("betaTimesS0", 0.3, 1),
            "I0": hp.choice("I0", range(15)),
            "Q0": hp.choice("Q0", range(5)),
            "start": start if start is not None else hp.choice("start", range(40, 60)),  # 疫情开始传播
            "hStart": hStart if hStart is not None else hp.choice("hStart", range(45, 60)),  # 开始实行隔离政策的天数
            "mStart": mStart if mStart is not None else hp.choice("mStart", range(80, 85)),  # 开始医疗条件升级的天数
            "mEnd": mEnd if mEnd is not None else hp.choice("mEnd", range(85, 90)),  # 开始医疗条件升级的天数
            "k": hp.uniform("k", 0.6, 0.7),  # 潜伏期患者与已发病患者传染强度的比率
            "epsilon": hp.uniform("epsilon", 0, 0.3),  # 潜伏期感染者转入未隔离感染者的比例
            "lamb": hp.uniform("lamb", 0.15, 0.21),  # 潜伏期感染者转入已隔离感染者的比例

            "theta": hp.uniform("theta", 0.1, 0.17),  # 未隔离感人者转入正在治疗的人群的比例
            "sigma": hp.uniform("sigma", 0., 0.35),  # 医疗资源扩充前，已隔离感人者转入正在治疗的人群的比例
            "gamma0": hp.uniform("gamma0", 0, 0.02),  # 医疗资源扩充后，正在治疗的已感染者治愈的比例
            "gamma1": hp.uniform("gamma1", 0.12, 0.22),  # 正在治疗的已感染者治愈的比例
            "delta": hp.uniform("delta", 0.01, 0.1),  # 未隔离的已感染者的死亡率
            "eta": hp.uniform("eta", 0, 0.01),  # 接受治疗的病人的死亡率
        }





    def opt_medical_upgrade(self, max_evals):
        """
        调用hyperopt库的API接口在搜索空间中确定最佳超参数优化总距离（成本）
        :param max_evals:
        :return:
        """
        def medical_upgrade_objective(p):
            """
            模型超参数搜索优化总距离（成本）
            :param p:
            :return:
            """
            seir_solution = get_medical_upgrade_solution(p)
            # loss = (np.abs(DATA_JRD - seir_solution)).mean()
            loss = np.abs(np.diff(seir_solution, axis=0) - self.DATA_JRD_DIFF).mean()
            # loss = ((np.diff(seir_solution, axis=0)-DATA_JRD_DIFF)**2).mean()
            result = {"total_loss": loss}
            return {'loss': result["total_loss"], "result": result, 'parameters': p, 'status': STATUS_OK}
        trials = Trials()
        best = fmin(fn=medical_upgrade_objective, space=self.space, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials)
        return best, trials


    def get_best_parameters_for_medical_upgrade(self):
        """
        封装opt_journey解析参数搜索结果
        :param max_evals:
        :return:
        """
        best, trials = self.opt_medical_upgrade(self.max_evals)
        # Sort the trials with lowest loss first
        trials_list = sorted(trials.results, key=lambda x: x['loss'])
        bestParameters = trials_list[0]['parameters']
        result = trials_list[0]["result"]
        # print(trials_list[0])
        print("best parameter found after {} trials".format(len(trials_list)))
        print(bestParameters)
        print("lowest loss is")
        print(result)
        plot.plot_trials(trials_list, name=self.province)
        loss_step = [step["loss"] for step in trials.results]
        plot.plot_loss_time(loss_step, name=self.province)

        seir_solution = get_medical_upgrade_solution(bestParameters)
        plot.compare_result_with_data(seir_solution, self.DATA_JRD, province=self.province)
        test_SEIR(bestParameters, name=self.province)
        return bestParameters

