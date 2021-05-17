import numpy as np
import pandas as pd
import gc
import sys

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb

import plot
import read_data


def LGBM(data, target, N_FOLDS=5):
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)
    # Training set
    train_set = lgb.Dataset(X_train, label=y_train)
    test_set = lgb.Dataset(X_test, label=y_test)

    model = lgb.LGBMClassifier(random_state=50)
    # Default hyperparamters
    hyperparameters = model.get_params()
    print(hyperparameters)

    model.fit(X_train, y_train)
    preds = np.argmax(model.predict_proba(X_test), axis=1)
    baseline = metrics.accuracy_score(y_test, preds)
    print('The baseline model {} scores {:.5f} accuracy_score on the test set.'.format(
        sys._getframe().f_code.co_name, baseline))


def randomForest(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

    # ros = SMOTE(k_neighbors=3)
    # X_train, y_train = ros.fit_sample(X_train, y_train)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    baseline = metrics.accuracy_score(y_test, y_predict)
    print('The baseline model {} scores {:.5f} accuracy_score on the test set.'.format(
        sys._getframe().f_code.co_name, baseline))
    # print(metrics.classification_report(y_test, y_predict))
    # print(metrics.confusion_matrix(y_test, y_predict))


def sensitivity_analysis(model, X, feature_name, win_size, step_size):
    feature = np.array(X[feature_name])
    steps = np.arange(-win_size, win_size, step_size)
    deltas = np.zeros((X.shape[0], len(steps)))
    result = np.zeros((X.shape[0], len(steps)))
    for i, delta in enumerate(steps):
        deltas[:, i] = delta
        X[feature_name] = feature + delta
        result[:, i] = model.predict(X)
    plot.plot_regression_sensitivity(deltas, result, name=feature_name)


def XGBoostRegressor(data, target, test_features):
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size=500)

    # xgboost模型训练
    model = xgb.XGBRegressor(
        learning_rate=0.3,
        n_estimators=100,
        max_depth=6,
        min_child_weight=2,
        gamma=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        n_jobs=-1)
    model.fit(X_train, y_train)
    # digraph = xgb.to_graphviz(model)
    # digraph.format = 'png'
    # digraph.view('../img/xgbtree')
    y_pred = model.predict(X_test)
    print(y_pred.shape)

    # 计算准确率
    baseline = metrics.accuracy_score(y_test, (y_pred+0.5).astype(np.int))
    print('The baseline model {} scores {:.5f} accuracy_score on the test set.'.format(
        sys._getframe().f_code.co_name, baseline))
    # plot.plot_xgb_fscore(model)
    gain = pd.DataFrame.from_dict(model._Booster.get_score(importance_type="gain"), orient='index')
    weight = pd.DataFrame.from_dict(model._Booster.get_score(importance_type="weight"), orient='index')
    cover = pd.DataFrame.from_dict(model._Booster.get_score(importance_type="cover"), orient='index')
    importance = pd.concat((gain, weight, cover), axis=1)
    importance.columns = ["gain", "weight", "cover"]
    importance = importance/importance.sum()
    importance.to_excel("../cache/importance.xlsx")
    plot.plot_xgb_fscore(importance)
    # print(importance)
    num_analysis = 10
    sensitivity_analysis(model, X_test[:num_analysis], "alcohol", .5, 0.05)
    sensitivity_analysis(model, X_test[:num_analysis], "residual_sugar", .5, 0.05)
    sensitivity_analysis(model, X_test[:num_analysis], "density", .5, 0.05)
    sensitivity_analysis(model, X_test[:num_analysis], "total_sulfur_dioxide", .5, 0.05)



def XGBoost_cv(data, target, test_features):
    # plot.plot_labels_class_counts(target, name="data")
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, stratify=target, test_size=800)

    smote = SMOTE(k_neighbors=3)
    X_train, y_train = smote.fit_sample(X_train, y_train)
    X_train = pd.DataFrame(X_train, columns=X_test.columns)
    # xgboost模型训练
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=10,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        scale_pos_weight=1,
        n_jobs=-1)
    model.fit(X_train, y_train)
    # digraph = xgb.to_graphviz(model)
    # digraph.format = 'png'
    # digraph.view('../img/xgbtree')
    y_pred = model.predict_proba(X_test)
    print(y_pred.shape)
    y_pred = np.argmax(y_pred, axis=1)
    baseline = metrics.accuracy_score(y_test, y_pred)
    print('The baseline model {} scores {:.5f} accuracy_score on the test set.'.format(
        sys._getframe().f_code.co_name, baseline))

    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame.from_dict(report)[["micro avg", "macro avg", "weighted avg"]]
    report.columns = ["微平均", "宏平均", "加权平均"]
    report.to_excel("../cache/classification_report.xlsx")
    print(report)
    CM = metrics.confusion_matrix(y_test, y_pred)
    plot.plot_CM(CM, name="problem1")
    test_pred = model.predict(test_features)
    plot.plot_labels_class_counts(test_pred, name="test_pred")
    num_analysis = 30
    sensitivity_analysis(model, X_test[:num_analysis], "alcohol", .5, 0.05)
    sensitivity_analysis(model, X_test[:num_analysis], "residual_sugar", .5, 0.05)
    sensitivity_analysis(model, X_test[:num_analysis], "density", .5, 0.05)
    sensitivity_analysis(model, X_test[:num_analysis], "total_sulfur_dioxide", .5, 0.05)

    gain = pd.DataFrame.from_dict(model._Booster.get_score(importance_type="gain"), orient='index')
    weight = pd.DataFrame.from_dict(model._Booster.get_score(importance_type="weight"), orient='index')
    cover = pd.DataFrame.from_dict(model._Booster.get_score(importance_type="cover"), orient='index')
    importance = pd.concat((gain, weight, cover), axis=1)
    importance.columns = ["gain", "weight", "cover"]
    importance = importance/importance.sum()
    importance.to_excel("../cache/importance.xlsx")
    plot.plot_xgb_fscore(importance)



if __name__ == '__main__':
    df, tgt, test_features = read_data.read_data()
    # LGBM(df, tgt)
    # randomForest(df, tgt)
    XGBoost_cv(df, tgt, test_features)
    # print('Baseline metrics')
    # print(metrics)

