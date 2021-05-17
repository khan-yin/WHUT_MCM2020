
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from common import *


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
    r, c = 2, 5
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
        pdf = PdfPages("..//img//"+name+"_trials.pdf")
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
    plt.xlabel("试验次数")
    plt.ylabel("log(损失函数)")

    if save:
        pdf = PdfPages("..//img//{}_loss_time.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def plot_provinceDF(provinceDF, country, save=True):
    def cm(i):
        return ["royalblue", "orange", "r"][int(i*2)]
    fig, axs = plt.subplots(2, 1, tight_layout=True)

    sampleSize = DAYS
    provinceDF_sampled = provinceDF[["confirmed", "recoveries", "deaths"]][
                         :sampleSize]  # .resample(rule='{}D'.format(WINDOW)).mean()
    provinceDF_sampled.plot(ax=axs[0], xlabel="2020.1.22起的天数", ylabel="人数", colormap=cm)
    provinceDF_sampled = provinceDF[["confirmedMA", "recoveriesMA", "deathsMA"]][:sampleSize]#.resample(rule='{}D'.format(WINDOW)).mean()
    provinceDF_sampled.plot(ax=axs[0], dashes=[6, 2], colormap=cm)
    axs[0].title.set_text('{} 累计确诊，死亡，康复病例'.format(country))

    provinceDF_sampled = provinceDF[["confirmedDaily", "recoveriesDaily", "deathsDaily"]][
                         :sampleSize]  # .resample(rule='{}D'.format(WINDOW)).mean()
    provinceDF_sampled.plot(ax=axs[1], xlabel="2020.1.22起的天数", ylabel="人数", colormap=cm)
    provinceDF_sampled = provinceDF[["confirmedDailyMA", "recoveriesDailyMA", "deathsDailyMA"]][:sampleSize]#.resample(rule='{}D'.format(WINDOW)).mean()
    provinceDF_sampled.plot(ax=axs[1], dashes=[6, 2], colormap=cm)
    axs[1].title.set_text('{} 新增确诊，死亡，康复病例'.format(country))

    plt.tight_layout()
    if save:
        pdf = PdfPages("../img/{}_provinceDF.pdf".format(country))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()


def compare_result_with_data(seir_solution, data, save=True, province=None):
    N_points = len(data)
    x = list(range(N_points))

    fig, axs = plt.subplots(2, 1, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].plot(x, data[:, 0], label="累计确诊病例", c="royalblue")
    axs[0].plot(x, data[:, 1], label="累计康复病例", c="orange")
    axs[0].plot(x, data[:, 2], label="累计死亡病例", c="r")

    axs[0].plot(x, seir_solution[:, 0], label="预测确诊病例", dashes=[6, 2], c="royalblue")
    axs[0].plot(x, seir_solution[:, 1], label="预测康复病例", dashes=[6, 2], c="orange")
    axs[0].plot(x, seir_solution[:, 2], label="预测死亡病例", dashes=[6, 2], c="r")
    axs[0].legend()

    x = x[:-1]
    data = np.diff(data, axis=0)
    seir_solution = np.diff(seir_solution, axis=0)
    axs[1].plot(x, data[:, 0], label="新增确诊病例", c="royalblue")
    axs[1].plot(x, data[:, 1], label="新增康复病例", c="orange")
    axs[1].plot(x, data[:, 2], label="新增死亡病例", c="r")

    axs[1].plot(x, seir_solution[:, 0], label="预测新增确诊", dashes=[6, 2], c="royalblue")
    axs[1].plot(x, seir_solution[:, 1], label="预测新增康复", dashes=[6, 2], c="orange")
    axs[1].plot(x, seir_solution[:, 2], label="预测新增死亡", dashes=[6, 2], c="r")
    axs[1].legend()
    if save:
        pdf = PdfPages("../img/{}_compareResult.pdf".format(province))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()


def compare_results(seir_solutions, paraName, paraLabel, paraVals, save=True, province=None):
    if province is None:
        province = PROVINCE
    N_points = len(seir_solutions[0])
    x = list(range(N_points))

    fig, axs = plt.subplots(2, 1, tight_layout=True)

    for i, seir_solution in enumerate(seir_solutions):
        # 新增确诊
        axs[0].plot(x, seir_solution[:, 0], label="{} = {}".format(paraLabel, paraVals[i]))
        # 新增死亡
        axs[1].plot(x, seir_solution[:, -1], label="{} = {}".format(paraLabel, paraVals[i]))
    axs[0].title.set_text("新增确诊")
    axs[0].legend()
    axs[0].grid()
    axs[1].title.set_text("新增死亡")
    axs[1].legend()
    axs[1].grid()
    if save:
        pdf = PdfPages("../img/{}_{}_sensitivity.pdf".format(province, paraName))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
