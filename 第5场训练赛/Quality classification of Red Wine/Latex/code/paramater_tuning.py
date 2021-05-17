from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import read_data
import plot
from common import *

from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import SMOTE


class ParameterTunung():
    def __init__(self, data, target, max_evals,
                 learning_rate=None, n_estimators=None, max_depth=None,
                 min_child_weight=None, gamma=None, over_sample=None,
                 colsample_bytree=None, objective=None):
        self.max_evals = max_evals
        X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, stratify=target, test_size=500)

        smote = SMOTE(k_neighbors=3)
        X_train_os, y_train_os = smote.fit_sample(X_train, y_train)
        self.DATA = (X_train, X_test, y_train, y_test, X_train_os, y_train_os)

        self.space = {
            "learning_rate": hp.uniform("learning_rate", 0.25, 0.5) if learning_rate is None else learning_rate,
            "n_estimators": hp.choice("n_estimators", range(100, 250)) if n_estimators is None else n_estimators,
            "max_depth": hp.choice("max_depth", range(5, 10)) if max_depth is None else max_depth,
            "min_child_weight": hp.choice("min_child_weight", range(1, 6)) if min_child_weight is None else min_child_weight,
            "gamma": hp.uniform("gamma", 0.1, 0.3) if gamma is None else gamma,
             # "subsample": hp.uniform("subsample", 0.6, 0.9) if subsample is None else subsample,
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.7) if colsample_bytree is None else colsample_bytree,
            "over_sample": hp.choice("over_sample", range(1)) if over_sample is None else over_sample,
            "regression": hp.choice("regression", range(2)) if objective is None else objective,
        }
        self.bestParameters = None

    def get_model(self, p):
        # xgboost模型训练
        if p["regression"]:
            model = xgb.XGBRegressor(
                learning_rate=p['learning_rate'],
                n_estimators=p['n_estimators'],
                max_depth=p['max_depth'],
                min_child_weight=p['min_child_weight'],
                gamma=p['gamma'],
                colsample_bytree=p['colsample_bytree'],
                tree_method='gpu_hist',
                object="reg:squarederror",
                verbosity=0,
                n_jobs=-1)
        else:
            model = xgb.XGBClassifier(
                learning_rate=p['learning_rate'],
                n_estimators=p['n_estimators'],
                max_depth=p['max_depth'],
                min_child_weight=p['min_child_weight'],
                gamma=p['gamma'],
                colsample_bytree=p['colsample_bytree'],
                tree_method='gpu_hist',
                verbosity=0,
                n_jobs=-1)
        if p["over_sample"]:
            model.fit(self.DATA[4], self.DATA[5])
        else:
            model.fit(self.DATA[0], self.DATA[2])
        return model

    def opt_parameters(self):
        """
        调用hyperopt库的API接口在搜索空间中确定最佳超参数
        :param max_evals:
        :return:
        """
        def _objective(p):
            """
            模型超参数搜索优化总距离（成本）
            :param p:
            :return:
            """
            model = self.get_model(p)
            y_pred = model.predict(self.DATA[1])
            if p["regression"]:
                y_pred = (y_pred+0.5).astype(np.int)
            loss = 1-metrics.accuracy_score(self.DATA[3], y_pred)
            result = {"total_loss": loss}
            return {'loss': result["total_loss"], "result": result, 'parameters': p, 'status': STATUS_OK}
        trials = Trials()
        best = fmin(fn=_objective, space=self.space, algo=tpe.suggest,
                    max_evals=self.max_evals, trials=trials)
        return best, trials

    def get_best_parameters(self):
        """
        封装opt_journey解析参数搜索结果
        :param max_evals:
        :return:
        """
        best, trials = self.opt_parameters()
        # Sort the trials with lowest loss first
        trials_list = sorted(trials.results, key=lambda x: x['loss'])
        bestParameters = trials_list[0]['parameters']
        result = trials_list[0]["result"]
        # print(trials_list[0])
        print("best parameter found after {} trials".format(len(trials_list)))
        print(bestParameters)
        print("lowest loss is")
        print(result)
        plot.plot_trials(trials_list)
        loss_step = [step["loss"] for step in trials.results]
        plot.plot_loss_time(loss_step)
        self.bestParameters = bestParameters
        return bestParameters

    def eval_best_model(self, p=None):
        if self.bestParameters is None and p is None:
            raise Exception("Have not called get_best_parameters")
        if p is not None and self.bestParameters is None:
            print("Using given Parameters")
            self.bestParameters = p
        best_model = self.get_model(self.bestParameters)
        # 计算准确率
        y_pred = best_model.predict(self.DATA[1])
        y_test = self.DATA[3]
        if self.bestParameters["regression"]:
            y_pred = (y_pred+0.5).astype(np.int)
        baseline = metrics.accuracy_score(y_test, y_pred)
        print('The baseline model XGBoost scores {:.5f} accuracy_score on the test set.'.format(baseline))
        CM = metrics.confusion_matrix(y_test, y_pred)
        plot.plot_CM(CM, name="problem3")

        report = metrics.classification_report(y_test, y_pred, output_dict=True)
        report = pd.DataFrame.from_dict(report)[["micro avg", "macro avg", "weighted avg"]]
        report.columns = ["微平均", "宏平均", "加权平均"]
        report.to_excel("../cache/classification_report_problem3.xlsx")



if __name__ == '__main__':
    df, tgt, test_features = read_data.read_data()
    pt = ParameterTunung(df, tgt, max_evals=50)
    # bestParameters = pt.get_best_parameters()
    bestParameters = bestParametersDict["0.316"]
    pt.eval_best_model(bestParameters)

