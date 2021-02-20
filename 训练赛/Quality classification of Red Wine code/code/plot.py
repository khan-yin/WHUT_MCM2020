
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from common import *

import xgboost as xgb


def plot_mat(mat, save=True, name="mat"):
    plt.figure()
    plt.imshow(mat, cmap=plt.cm.hot,
               # vmin=0, vmax=1
               )
    plt.colorbar()
    if save:
        pdf = PdfPages("img//"+name+".pdf")
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def get_trials_df(trials, hypers):
    df = pd.DataFrame(columns=hypers+["loss"])
    for trial in trials:
        row_dict = trial['parameters'].copy()
        row_dict["loss"] = trial["loss"]
        df = df.append(row_dict, ignore_index=True)
    return df


def plot_trials(trials, save=True, name=""):
    r, c = 2, 3
    hypers = list(trials[0]['parameters'].keys())[:r*c]
    trials_df = get_trials_df(trials, hypers)
    best_trial = trials_df.iloc[0, :]
    # assert len(hypers) == r*c
    fig, axs = plt.subplots(r, c, figsize=(4*c, 4*r))
    for i in range(r):
        for j in range(c):
            hyper = hypers[c*i+j]
            sns.regplot(hyper, "loss", data=trials_df[[hyper, "loss"]], ax=axs[i, j])
            axs[i, j].scatter(best_trial[hyper], best_trial["loss"], marker='*', s=200, c='k')
            axs[i, j].set(xlabel='{}'.format(hyper), ylabel="loss")
    plt.tight_layout()
    if save:
        pdf = PdfPages(".//img//"+name+"_trials.pdf")
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_loss_time(loss_step, save=True, name=""):
    loss_time = []
    bestLoss = loss_step[0]
    for loss in loss_step:
        if loss < bestLoss:
            loss_time.append(loss)
            bestLoss = loss
        else:
            loss_time.append(bestLoss)
    plt.figure()
    plt.plot(list(range(len(loss_time))), np.log(np.array(loss_time)))
    plt.xlabel("Training times")
    plt.ylabel("loss")

    if save:
        pdf = PdfPages(".//img//{}_loss_time.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def plot_xgb_fscore(importance, save=True, name="xgb"):
    importance.sort_values(by="cover", axis=0, ascending=False, inplace=True)
    importance_types = importance.columns
    print(importance_types)
    x = range(len(importance.index))
    plt.bar(x,
            importance[importance_types[0]],
            label=importance_types[0])
    plt.bar(x,
            importance[importance_types[1]],
            label=importance_types[1],
            bottom=importance[importance_types[0]])
    plt.bar(x,
            importance[importance_types[2]],
            label=importance_types[2],
            bottom=importance[importance_types[0]]+importance[importance_types[1]])
    plt.legend()
    plt.ylabel('Feature Importance')
    plt.xticks(x, importance.index)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save:
        pdf = PdfPages(".//img//feature_importance_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()

    plt.close()
    plt.figure()
    cover = importance[importance_types[2]]
    plt.bar(x, cover, label=importance_types[2],)
    plt.legend()
    plt.ylabel('Feature Importance')
    plt.xticks(x, importance.index)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img/feature_importance_{}.pdf".format("cover"))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_corr(save=True, name=""):
    df = pd.read_excel("./data/final.csv", sheet_name=0).replace("N", np.nan)
    f, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="Reds", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle('heatmap', fontsize=14)
    if save:
        pdf = PdfPages(".//img//{}_corr.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_CM(CM, save=True, name=""):
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(CM, annot=True, ax=ax, fmt='.2f', linewidths=.05)
    f.subplots_adjust(top=0.93)
    f.suptitle('Confusion Matrix', fontsize=14)
    plt.xlabel("predict label")
    plt.ylabel("Positive label")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    if save:
        pdf = PdfPages(".//img//{}_CM.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_features(save=True, name=""):
    df = pd.read_excel("../data/final.csv", sheet_name=0).replace("N", np.nan)
    df.fillna(df.mean(), inplace=True)
    # df = (df-df.mean())/(df.std())
    assert not np.isnan(df).any().any()

    r, c = 4, 3
    sns.set()
    # sns.set()
    # sns.relplot(data=df, x='alcohol', y="quality", kind='line', height=5, aspect=2, color='red')
    # plt.show()
    cols = list(df.columns)
    # assert len(hypers) == r*c
    fig, axs = plt.subplots(r, c, figsize=(4*c, 4*r))
    for i in range(r):
        for j in range(c):
            if c*i+j >= len(cols):
                continue
            col = cols[c*i+j]
            sns.regplot(data=df, x=col, y="quality", ax=axs[i, j])
            # axs[i, j].scatter(df[col], df["quality"])

    plt.tight_layout()
    if save:
        plt.savefig(".//img//"+name+"_features.png")
        pdf = PdfPages(".//img//"+name+"_features.pdf")
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def draw_multivarient_plot(dataset=None, rows=4, cols=3, plot_type="violin", save=True, name=""):
    if dataset is None:
        dataset = pd.read_excel("../data/final.csv", sheet_name=0).replace("N", np.nan)
    # Veri setindeki sütünların isimleri alınıyor
    column_names = dataset.columns.values
    # Kaç tane sütün olduğu bulunuyor
    number_of_column = len(column_names)

    # Satır*sütün boyutlarında alt grafik içeren
    # matris oluşturuluyor. Matrisin genişliği:22 yüksekliği:16
    fig, axarr = plt.subplots(rows, cols, figsize=(22, 16))

    counter = 0  # Çizimi yapılacak özelliğin column_names listesindeki indeks değerini tutuyor
    for i in range(rows):
        for j in range(cols):
            if 'violin' in plot_type:
                sns.violinplot(x='quality', y=column_names[counter], data=dataset, ax=axarr[i][j])
            elif 'box' in plot_type:
                sns.boxplot(x='quality', y=column_names[counter], data=dataset, ax=axarr[i][j])
            elif 'point' in plot_type:
                sns.pointplot(x='quality', y=column_names[counter], data=dataset, ax=axarr[i][j])
            elif 'bar' in plot_type:
                sns.barplot(x='quality', y=column_names[counter], data=dataset, ax=axarr[i][j])

            counter += 1
            if counter == (number_of_column - 1,):
                break
    plt.tight_layout()
    if save:
        pdf = PdfPages(".//img//" + name + "{}_multivarient.pdf".format(plot_type))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def plot_regression_sensitivity(deltas, result, save=True, name=""):
    plt.figure()
    for i in range(len(deltas)):
        plt.plot(deltas[i], result[i], dashes=[6, 2])
    result_mean = np.mean(result, axis=0)
    # plt.scatter(deltas[0], result_mean, marker="s")
    sns.regplot(deltas[0], result_mean, color="red")
    plt.title(name+"Sensitivity Analysis")
    plt.xlabel(name+"value")
    plt.ylabel("Quality")
    plt.tight_layout()
    if save:
        pdf = PdfPages(".//img//sensitivity_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def autolabel(rects, ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


def plot_labels_class_counts(labels, save=True, name=""):
    labels += 3
    x = np.array([3, 4, 5, 6, 7, 9, 8])
    num = len(labels)
    class_counts = []
    for tgt in x:
        class_counts.append((labels == tgt).sum())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    rects1 = axs[0].bar(x, class_counts)
    autolabel(rects1, axs[0])
    axs[0].set_xlabel("predict label")
    axs[0].set_ylabel("Sample counts")
    wedges, _ = axs[1].pie(class_counts, explode=np.zeros_like(x)+0.1, shadow=True)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    recipe = ["label{}, {:.1f}%".format(x[i], 100*class_counts[i]/num) for i in range(len(x))]
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        axs[1].annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    plt.tight_layout()
    if save:
        pdf = PdfPages(".//img//labels_class_counts_{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    plot_xgb_fscore()
    # draw_multivarient_plot(plot_type="box")
