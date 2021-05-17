import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import shutil

IGNORE_EARLIER_REPORT = 30
img_dataset_path = './data/2021MCMProblemC_DataSet.xlsx'
img_id_path = './data/2021MCM_ProblemC_ Images_by_GlobalID.xlsx'


def get_hasimg_csv():
    try:
        hasimg_csv = pd.read_csv('./data/hasimg.csv')
        hasimg_csv['Detection Date'] = pd.to_datetime(hasimg_csv['Detection Date'])
        return hasimg_csv
    except FileNotFoundError:
        img_id_csv = pd.read_excel(img_id_path)
        img_id_csv = img_id_csv[['FileName', 'GlobalID']]
        img_dataset = pd.read_excel(img_dataset_path)
        img_dataset.head()
        pd.Timestamp.min.ceil('D')
        img_dataset = img_dataset[img_dataset['Detection Date'] != '<Null>']
        img_dataset['Detection Date'] = pd.to_datetime(img_dataset['Detection Date'], errors="coerce")
        img_dataset = img_dataset.dropna(subset=['Detection Date'])
        img_dataset['Detection Date']
        img_ds_idindex = img_dataset.set_index('GlobalID')
        img_id_idindex = img_id_csv.set_index('GlobalID')
        hasimg_csv = img_ds_idindex.join(img_id_idindex, how='left')
        hasimg_csv = hasimg_csv.reset_index()
        hasimg_csv = hasimg_csv.drop_duplicates('GlobalID')#去除重复
        cls_mapping = {
            'Negative ID': 0,
            'Positive ID': 1,
            'Unprocessed': 2,
            'Unverified': 2
        }

        hasimg_csv['Lab Status']=hasimg_csv['Lab Status'].map(cls_mapping)
        hasimg_csv = hasimg_csv.sort_values(by='Detection Date')
        hasimg_csv.to_csv('./data/hasimg.csv')
        return hasimg_csv

{'0': {'precision': 0.9966261808367072, 'recall': 0.991275167785235, 'f1-score': 0.9939434724091522, 'support': 1490},
 '1': {'precision': 0.2777777777777778, 'recall': 0.5, 'f1-score': 0.35714285714285715, 'support': 10},
 'accuracy': 0.988, 'macro avg': {'precision': 0.6372019793072425, 'recall': 0.7456375838926175, 'f1-score': 0.6755431647760046, 'support': 1500},
 'weighted avg': {'precision': 0.9918338581496476, 'recall': 0.988, 'f1-score': 0.9896981349740436, 'support': 1500}}
def plot_maps(hasimg_csv, n=12, timespan=None, save=True, name="maps"):
    # print(station["classID"])
    latest=hasimg_csv["Detection Date"].iloc[-1]
    if timespan is None:
        earliest = hasimg_csv["Detection Date"].iloc[0]
        timespan = latest-earliest
    else:
        earliest = latest-timespan
    timestep=timespan/n
    fig, axs = plt.subplots(n//3, 3, figsize=(9, 3*n/4))
    for i in range(n):
        begin = earliest+timestep*i
        end = earliest+timestep*(i+1)
        time_silce=hasimg_csv[(hasimg_csv["Detection Date"]>begin) & (hasimg_csv["Detection Date"]<end)]
        positive_csv = time_silce[time_silce['Lab Status']==1]
        negative_csv = time_silce[time_silce['Lab Status']==0]
        unverified_csv = time_silce[time_silce['Lab Status']==2]
        axs[i//3, i%3].scatter(negative_csv['Longitude'], negative_csv['Latitude'],c='b',alpha=.5)
        axs[i//3, i%3].scatter(unverified_csv['Longitude'], unverified_csv['Latitude'],c='orange',alpha=.5)
        axs[i//3, i%3].scatter(positive_csv['Longitude'], positive_csv['Latitude'],c='r')
        axs[i//3, i%3].set_xlim((-125, -116))
        axs[i//3, i%3].set_ylim((45, 50))
        axs[i//3, i%3].set_xlabel("Longitude")
        axs[i//3, i%3].set_ylabel("Latitude")
        axs[i//3, i%3].set_title('{} to {}'.format(begin.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    plt.title("Report occurred within a time period")
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        plt.show()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_report_kde(hasimg_csv, save=True, name="report_kde"):
    hasimg_csv = hasimg_csv[hasimg_csv["Detection Date"] > pd.Timestamp('2010-01-01')]
    date_sq = hasimg_csv["Detection Date"]#.iloc[IGNORE_EARLIER_REPORT:]
    day_sq = (date_sq - date_sq.iloc[0]) / pd.Timedelta(days=1)
    day_sq.name = 'All report'
    ax = sns.kdeplot(day_sq, shade=True, color='gray')

    date_sq = pd.Series(pd.date_range(date_sq.iloc[0], date_sq.iloc[-1]))

    date_ls = list(date_sq.dt.strftime('%Y-%m-%d'))
    ax.set_xlim((0, (date_sq.iloc[-1] - date_sq.iloc[0]) / pd.Timedelta(days=1)))
    xticks = ax.get_xticks()
    step_len = len(date_ls)//(len(xticks)-1)

    ax.set_xticklabels([date_ls[i*step_len] for i in range(len(xticks))])
    print(list(ax.get_xticklabels()))
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability density of the report")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_report_kdes(hasimg_csv, save=True, name="report_kdes"):
    date_sq = hasimg_csv[hasimg_csv["Detection Date"] > pd.Timestamp('2010-01-01')]["Detection Date"]
    date_sq_2019 = hasimg_csv[hasimg_csv["Detection Date"] > pd.Timestamp('2019-01-01')]["Detection Date"]
    date_sq.name = 'All report'
    date_sq_2019.name = 'All report'

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_xlim((0, (date_sq.iloc[-1] - date_sq.iloc[0]) / pd.Timedelta(days=1)))
    sns.kdeplot((date_sq-date_sq.iloc[0])/pd.Timedelta(days=1), shade=True, ax=axs[0]
                 #, color='r'
                )
    axs[1].set_xlim((0, (date_sq_2019.iloc[-1] - date_sq_2019.iloc[0]) / pd.Timedelta(days=1)))
    sns.kdeplot((date_sq_2019-date_sq_2019.iloc[0])/pd.Timedelta(days=1), shade=True, ax=axs[1]
                 #, color='b'
                )
    date_sq = pd.Series(pd.date_range(date_sq.iloc[0], date_sq.iloc[-1]))
    date_ls = list(date_sq.dt.strftime('%Y-%m-%d'))
    xticks = axs[0].get_xticks()
    step_len = len(date_ls)//len(xticks)
    xticklabels = [date_ls[i*step_len] for i in range(len(xticks))]
    axs[0].set_xticklabels(xticklabels, rotation=45)
    axs[0].set_title("Density of all report from 2010 to present")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Probability density of the report")

    date_sq_2019 = pd.Series(pd.date_range(date_sq_2019.iloc[0], date_sq.iloc[-1]))
    date_ls_2019 = list(date_sq_2019.dt.strftime('%Y-%m-%d'))
    xticks = axs[1].get_xticks()
    step_len = len(date_ls_2019) // len(xticks)
    xticklabels = [date_ls_2019[i*step_len] for i in range(len(xticks))]
    axs[1].set_xticklabels(xticklabels, rotation=45)
    axs[1].set_title("Density of all report from 2019 to present")
    axs[1].set_xlabel("Date")
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_combined_report_kdes(hasimg_csv, save=True, name="combined_report_kdes"):
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    hasimg_csv = hasimg_csv.iloc[IGNORE_EARLIER_REPORT:]
    date_sq = hasimg_csv["Detection Date"]
    day_sq = (date_sq - date_sq.iloc[0]) / pd.Timedelta(days=1)
    day_sq.name = 'All report'
    sns.kdeplot(day_sq, shade=True, color='gray', ax=axs[0])

    earliest = date_sq.iloc[0]

    positive_csv = hasimg_csv[hasimg_csv['Lab Status']==1]
    negative_csv = hasimg_csv[hasimg_csv['Lab Status']==0]
    unverified_csv = hasimg_csv[hasimg_csv['Lab Status']==2]
    pos_date_sq = positive_csv["Detection Date"]
    neg_date_sq = negative_csv["Detection Date"]
    unv_date_sq = unverified_csv["Detection Date"]
    pos_date_sq.name = "Positive report"
    neg_date_sq.name = "Negative report"
    unv_date_sq.name = "Unverified report"
    sns.kdeplot((pos_date_sq-earliest)/pd.Timedelta(days=1), shade=True, ax=axs[1]
                 , color='r'
                )
    sns.kdeplot((neg_date_sq-earliest)/pd.Timedelta(days=1), shade=True, ax=axs[1]
                 , color='royalblue'
                )
    sns.kdeplot((unv_date_sq-earliest)/pd.Timedelta(days=1), shade=True, ax=axs[1]
                 , color='orange'
                )
    date_sq = pd.Series(pd.date_range(date_sq.iloc[0], date_sq.iloc[-1]))
    date_ls = list(date_sq.dt.strftime('%Y-%m-%d'))
    xticks = axs[1].get_xticks()
    step_len = len(date_ls)//len(xticks)
    xticklabels = [date_ls[i*step_len] for i in range(len(xticks))]

    axs[0].set_xlim((0, (date_sq.iloc[-1] - date_sq.iloc[0]) / pd.Timedelta(days=1)))
    axs[1].set_xlim((0, (date_sq.iloc[-1] - date_sq.iloc[0]) / pd.Timedelta(days=1)))
    axs[1].set_xticklabels(xticklabels,
                           #rotation=45
                           )
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Probability density of the report")
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(rect.get_height())
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


START_DATE = pd.Timestamp('2019-05-01')
END_DATE = pd.Timestamp('2020-11-01')
def plot_report_bar(hasimg_csv, width=.35, save=True, name="report_bar"):
    #hasimg_csv = hasimg_csv.iloc[IGNORE_EARLIER_REPORT:]
    hasimg_csv = hasimg_csv[hasimg_csv["Detection Date"] > START_DATE]
    date_sq = hasimg_csv["Detection Date"]
    start_d = date_sq.iloc[0]
    end_d = date_sq.iloc[-1]

    positive_csv = hasimg_csv[hasimg_csv['Lab Status']==1]["Detection Date"]
    negative_csv = hasimg_csv[hasimg_csv['Lab Status']==0]["Detection Date"]
    unverified_csv = hasimg_csv[hasimg_csv['Lab Status']==2]["Detection Date"]
    pos_cnt = positive_csv.groupby([positive_csv.dt.year.rename('year'),
                                    positive_csv.dt.month.rename('month')]
                                   ).agg({'count'})
    neg_cnt = negative_csv.groupby([negative_csv.dt.year.rename('year'),
                                    negative_csv.dt.month.rename('month')]
                                   ).agg({'count'})
    unv_cnt = unverified_csv.groupby([unverified_csv.dt.year.rename('year'),
                                    unverified_csv.dt.month.rename('month')]
                                   ).agg({'count'})
    cnt = pd.concat((pos_cnt, neg_cnt, unv_cnt),
              axis=1).fillna(0)
    cnt.columns = ["Positive report", "Negative report","Unverified report"]
    fig, axs = plt.subplots(2, 1,
                            #sharex=True,
                            figsize=(8, 6))
    report_per_month = pd.DataFrame(index=pd.date_range(start_d, end_d, freq='M'))
    x = np.arange(len(cnt))

    rects1 = axs[0].bar(x, cnt["Positive report"], width, label="Positive report", color='r')
    rects2 = axs[1].bar(x - width/2, cnt["Negative report"], width, label="Negative report")
    rects3 = axs[1].bar(x + width/2, cnt["Unverified report"], width, label="Unverified report")


    autolabel(rects1, axs[0])
    autolabel(rects2, axs[1])
    autolabel(rects3, axs[1])
    axs[0].set_xticks(list(range(len(cnt.index))))
    axs[0].set_xticklabels(['{}-{}'.format(cnt.index[i][0], cnt.index[i][1]) for i in range(len(cnt))],
                           rotation=45
                           )
    axs[0].legend()
    axs[1].set_xticks(list(range(len(cnt.index))))
    axs[1].set_xticklabels(['{}-{}'.format(cnt.index[i][0], cnt.index[i][1]) for i in range(len(cnt))],
                           rotation=45
                           )
    axs[1].legend()
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_class_bar(hasimg_csv, width=.35, save=True, name="class_bar"):
    hasimg_csv = hasimg_csv[hasimg_csv["Detection Date"] > START_DATE]

    cnt = hasimg_csv.groupby('Lab Status').agg('count')["Detection Date"]
    cnt.index = ["Negative report", "Positive report", "Unverified report"]
    fig, axs = plt.subplots(figsize=(4, 6))
    rects1 = axs.bar(range(len(cnt)), cnt, width, label="Positive report")

    axs.set_xticks([0, 1, 2])
    axs.set_xticklabels(cnt.index,
                           #rotation=45
                           )
    autolabel(rects1, axs)
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


from sklearn import svm

num_levels = 25

def plot_species_svms(hasimg_csv, grid_size=0.1, norm=False, gamma=3,
                              save=True, name="species_svm"):
    xmin, xmax = (-125, -116)
    ymin, ymax = (45, 50)
    xgrid = np.arange(xmin, xmax, grid_size)
    ygrid = np.arange(ymin, ymax, grid_size)

    # The grid in x,y coordinates
    X, Y = np.meshgrid(xgrid, ygrid[::-1])
    # background points (grid coordinates) for evaluation
    np.random.seed(13)
    fig, axs = plt.subplots(2, 2, figsize=(6*9/5, 6))
    names = ['All reports', "Negative report", "Positive report", "Unverified report"]
    dark = 0.7
    cmap = mpl.colors.ListedColormap([[1.-(i/num_levels)*dark, 1.-(i/num_levels)*dark, 1.-(i/num_levels)*dark]for i in range(num_levels)])
    cmap.set_over((1., 1., 1.))
    cmap.set_under((0.5, 0.5, 0.5))
    cms = [cmap, plt.cm.Blues, plt.cm.Reds, plt.cm.Oranges]

    for i in range(4):
        # Fit, predict, and plot
        if i == 0:
            train_df = hasimg_csv[['Longitude', 'Latitude']]
        else:
            train_df = hasimg_csv[hasimg_csv['Lab Status']==i-1][['Longitude', 'Latitude']]
        train_cover_std = np.array(train_df)

        # Standardize features
        mean = train_cover_std.mean(axis=0)
        std = train_cover_std.std(axis=0)
        if norm:
            train_cover_std = (train_cover_std - mean) / std

        print(" - fit OneClassSVM ... ", end='')
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma)
        clf.fit(train_cover_std)
        print("done.")
        print(" - predict species distribution")

        Z = np.ones((ygrid.size, xgrid.size), dtype=np.float64)
        idx = np.meshgrid(np.arange(ygrid.size), np.arange(xgrid.size))
        idx[0], idx[1] = idx[0].flatten(), idx[1].flatten()
        coverages = np.array([X, Y])
        coverages_land = coverages[:, idx[0], idx[1]].T
        if norm:
            pred = clf.decision_function((coverages_land - mean) / std)
        else:
            pred = clf.decision_function(coverages_land)
        print("done.")
        # pred -= pred.min()
        # pred /= pred.max()
        Z[idx[0], idx[1]] = np.exp(pred)

        levels = np.linspace(Z.min(), Z.max(), num_levels)
        # plot contours of the prediction
        im = axs[i//2, i%2].contourf(X, Y, Z, levels=levels, cmap=cms[i])
        fig.colorbar(im, ax=axs[i//2, i%2], format='%.1f')
        axs[i//2, i%2].scatter(train_df['Longitude'], train_df['Latitude'],
                    s=1, c='black',
                    marker='^', label=names[i])
        axs[i//2, i%2].set_ylabel('Latitude')
        axs[i//2, i%2].set_xlabel('Longitude')
        #axs[i//2, i%2].legend()
        axs[i//2, i%2].set_title(names[i])
    #plt.title(name)
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_species_spread_svms(hasimg_csv, grid_size=0.1, norm=False, gamma=7, w=1,
                             start_date=pd.Timestamp('2019-09-01'), end_date=END_DATE,
                              save=True, name="species_spread_svm"):

    timestep = (end_date - start_date)/6
    xmin, xmax = (-125, -116)
    ymin, ymax = (45, 50)
    # x coordinates of the grid cells
    xgrid = np.arange(xmin, xmax, grid_size)
    # y coordinates of the grid cells
    ygrid = np.arange(ymin, ymax, grid_size)

    # The grid in x,y coordinates
    X, Y = np.meshgrid(xgrid, ygrid[::-1])

    # background points (grid coordinates) for evaluation
    np.random.seed(13)
    fig, axs = plt.subplots(2, 3, figsize=(9, 4))

    Z = np.zeros((ygrid.size, xgrid.size), dtype=np.float64)
    def get_kde(train_cover_std):
        # Standardize features
        mean = train_cover_std.mean(axis=0)
        std = train_cover_std.std(axis=0)
        if norm:
            train_cover_std = (train_cover_std - mean) / std

        # Fit OneClassSVM
        print(" - fit OneClassSVM ... ", end='')
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma)
        clf.fit(train_cover_std)
        print("done.")
        print(" - predict species distribution")

        deltaZ = np.zeros((ygrid.size, xgrid.size), dtype=np.float64)

        idx = np.meshgrid(np.arange(ygrid.size), np.arange(xgrid.size))
        idx[0], idx[1] = idx[0].flatten(), idx[1].flatten()
        coverages = np.array([X, Y])
        coverages_land = coverages[:, idx[0], idx[1]].T

        if norm:
            pred = clf.decision_function((coverages_land - mean) / std)
        else:
            pred = clf.decision_function(coverages_land)

        deltaZ[idx[0], idx[1]] = np.exp(pred)
        print("done.")
        return deltaZ
    for i in range(6):
        begin = start_date + timestep * i
        end = start_date + timestep * (i + 1)
        # Fit, predict, and plot
        train_df_pos = hasimg_csv[(hasimg_csv["Detection Date"]>begin) &
                              (hasimg_csv["Detection Date"]<end) &
                              (hasimg_csv["Lab Status"]==1)][['Longitude', 'Latitude']]
        if len(train_df_pos)==0:
            dZ = np.zeros((ygrid.size, xgrid.size), dtype=np.float64)
        else:
            Z_pos = get_kde(np.array(train_df_pos))
            Z_pos -= Z_pos.min()
            train_df_ver = hasimg_csv[(hasimg_csv["Detection Date"]>begin) &
                                  (hasimg_csv["Detection Date"]<end) &
                                  (hasimg_csv["Lab Status"]!=2)][['Longitude', 'Latitude']]
            Z_ver = get_kde(np.array(train_df_ver))
            dZ = Z_pos/Z_ver
        Z = (Z*w + dZ)/(1+w)
        levels = np.linspace(Z.min(), Z.max(), num_levels)

        # plot contours of the prediction
        im = axs[i//3, i%3].contourf(X, Y, Z, levels=levels, cmap=plt.cm.coolwarm)
        fig.colorbar(im, ax=axs[i//3, i%3], format='%.1f')

        axs[i//3, i%3].scatter(train_df_pos['Longitude'], train_df_pos['Latitude'],
                    s=1, c='black',
                    marker='^')
        axs[i//3, i%3].set_ylabel('Latitude')
        axs[i//3, i%3].set_xlabel('Longitude')
        #axs[i//3, i%3].legend()
        axs[i//3, i%3].set_title('{} to {}'.format(begin.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))

    #plt.title(name)
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
def plot_species_total_mu_roc(hasimg_csv, grid_size=0.1, norm=False, gamma=7,
                             start_date=pd.Timestamp('2019-09-01'), end_date=END_DATE,
                              save=True, name="species_total_mu_roc"):
    xmin, xmax = (-125, -116)
    ymin, ymax = (45, 50)
    xgrid = np.arange(xmin, xmax, grid_size)
    ygrid = np.arange(ymin, ymax, grid_size)
    # The grid in x,y coordinates
    X, Y = np.meshgrid(xgrid, ygrid[::-1])
    # background points (grid coordinates) for evaluation
    np.random.seed(13)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma)

    def get_kde(train_cover_std):
        # Standardize features
        mean = train_cover_std.mean(axis=0)
        std = train_cover_std.std(axis=0)
        if norm:
            train_cover_std = (train_cover_std - mean) / std

        # Fit OneClassSVM
        print(" - fit OneClassSVM ... ", end='')
        clf.fit(train_cover_std)
        print("done.")
        print(" - predict species distribution")

        deltaZ = np.zeros((ygrid.size, xgrid.size), dtype=np.float64)

        idx = np.meshgrid(np.arange(ygrid.size), np.arange(xgrid.size))
        idx[0], idx[1] = idx[0].flatten(), idx[1].flatten()
        coverages = np.array([X, Y])
        coverages_land = coverages[:, idx[0], idx[1]].T

        if norm:
            pred = clf.decision_function((coverages_land - mean) / std)
        else:
            pred = clf.decision_function(coverages_land)

        deltaZ[idx[0], idx[1]] = np.exp(pred)
        print("done.")
        return deltaZ

    data_df = hasimg_csv[(hasimg_csv["Detection Date"]>start_date) &
                         (hasimg_csv["Detection Date"]<end_date) &
                         (hasimg_csv["Lab Status"]!=2)
                         ][
        ['Longitude', 'Latitude', "Lab Status"]]
    train_df_pos = data_df[data_df["Lab Status"]==1][['Longitude', 'Latitude']]

    Z_pos = get_kde(np.array(train_df_pos))
    Z_pos -= Z_pos.min()
    train_df_neg = data_df[data_df["Lab Status"]==0][['Longitude', 'Latitude']]
    Z_neg = get_kde(np.array(train_df_neg))
    dZ = Z_pos/(Z_neg+Z_pos)
    Z = dZ
    levels = np.linspace(Z.min(), Z.max(), num_levels)
    # plot contours of the prediction
    im = axs[0].contourf(X, Y, Z, levels=levels, cmap=plt.cm.coolwarm)
    fig.colorbar(im, ax=axs[0], format='%.1f')

    axs[0].scatter(train_df_pos['Longitude'], train_df_pos['Latitude'], label='Positive report', s=2, c='black', marker='^')
    axs[0].set_ylabel('Latitude')
    axs[0].set_xlabel('Longitude')
    axs[0].set_title('{} to {}'.format(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    axs[0].legend(loc="lower right")

    cv = StratifiedKFold(n_splits=5)
    mean_fpr = np.linspace(0, 1, 100)
    cnt = 0
    # 画平均ROC曲线的两个参数
    mean_tpr = 0.0              # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 100)
    X = np.array(data_df[['Longitude', 'Latitude']])
    y = np.array(data_df[['Lab Status']])
    for i, (train, test) in enumerate(cv.split(X, y)):  # 利用模型划分数据集和目标变量 为一一对应的下标
        cnt += 1
        probas_ = clf.fit(X[train]).decision_function(X[test])  # 训练模型后预测每条样本得到两种结果的概率
        fpr, tpr, thresholds = roc_curve(y[test], probas_)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

        roc_auc = auc(fpr, tpr)  # 求auc面积
        axs[1].plot(fpr, tpr, lw=1, label='ROC fold {0} (area = {1:.2f})'.format(i, roc_auc))  # 画出当前分割数据的ROC曲线

    mean_tpr /= cnt  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    mean_auc = auc(mean_fpr, mean_tpr)

    axs[1].plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.2f})'.format(mean_auc), lw=2)

    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    axs[1].set_title('Receiver Operating Characteristic Curve')
    axs[1].legend(loc="lower right")

    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

from mpl_toolkits.mplot3d import Axes3D
def plot_species_3d_mu(hasimg_csv, grid_size=0.1, norm=False, gamma=7,
                             start_date=pd.Timestamp('2019-09-01'), end_date=END_DATE,
                              save=True, name="species_3d_mu"):
    xmin, xmax = (-125, -116)
    ymin, ymax = (45, 50)
    xgrid = np.arange(xmin, xmax, grid_size)
    ygrid = np.arange(ymin, ymax, grid_size)
    # The grid in x,y coordinates
    X, Y = np.meshgrid(xgrid, ygrid[::-1])
    # background points (grid coordinates) for evaluation
    np.random.seed(13)
    fig = plt.figure(figsize=(12, 4))
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma)

    def get_kde(train_cover_std):
        # Standardize features
        mean = train_cover_std.mean(axis=0)
        std = train_cover_std.std(axis=0)
        if norm:
            train_cover_std = (train_cover_std - mean) / std

        # Fit OneClassSVM
        print(" - fit OneClassSVM ... ", end='')
        clf.fit(train_cover_std)
        print("done.")
        print(" - predict species distribution")

        deltaZ = np.zeros((ygrid.size, xgrid.size), dtype=np.float64)

        idx = np.meshgrid(np.arange(ygrid.size), np.arange(xgrid.size))
        idx[0], idx[1] = idx[0].flatten(), idx[1].flatten()
        coverages = np.array([X, Y])
        coverages_land = coverages[:, idx[0], idx[1]].T

        if norm:
            pred = clf.decision_function((coverages_land - mean) / std)
        else:
            pred = clf.decision_function(coverages_land)

        deltaZ[idx[0], idx[1]] = np.exp(pred)
        print("done.")
        return deltaZ

    data_df = hasimg_csv[(hasimg_csv["Detection Date"]>start_date) &
                         (hasimg_csv["Detection Date"]<end_date) &
                         (hasimg_csv["Lab Status"]!=2)
                         ][
        ['Longitude', 'Latitude', "Lab Status"]]
    train_df_pos = data_df[data_df["Lab Status"]==1][['Longitude', 'Latitude']]

    Z_pos = get_kde(np.array(train_df_pos))
    Z_pos -= Z_pos.min()
    train_df_neg = data_df[data_df["Lab Status"]==0][['Longitude', 'Latitude']]
    Z_neg = get_kde(np.array(train_df_neg))
    dZ = Z_pos/(Z_neg+Z_pos)
    Z = dZ
    levels = np.linspace(Z.min(), Z.max(), num_levels)
    # plot contours of the prediction
    axs = []
    axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
    im = axs[0].plot_surface(X, Y, Z_pos, rstride=1, cstride=1, cmap=plt.cm.Reds, alpha=1.0)
    #im = axs[0].contourf(X, Y, Z, levels=levels, zdir='z', offset=-1, cmap=plt.cm.Reds)
    fig.colorbar(im, ax=axs[0], format='%.1f')

    im = axs[0].plot_surface(X, Y, Z_neg, rstride=1, cstride=1, cmap=plt.cm.Blues, alpha=.5)
    fig.colorbar(im, ax=axs[0], format='%.1f')
    axs[0].set_ylabel('Latitude')
    axs[0].set_xlabel('Longitude')
    axs[0].set_title('Positive and negative density')

    axs.append(fig.add_subplot(1, 2, 2, projection='3d'))
    im = axs[1].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=1.)
    fig.colorbar(im, ax=axs[1], format='%.1f')
    axs[1].set_ylabel('Latitude')
    axs[1].set_xlabel('Longitude')
    axs[1].set_title('$\mu_i(X)$')

    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        plt.show()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_mean_roc_auc_by_gamma(hasimg_csv, grid_size=0.1, norm=False, gammas=np.arange(1,17,3), n_splits=5,
                             start_date=pd.Timestamp('2019-09-01'), end_date=END_DATE,
                              save=True, name="mean_roc_auc_by_gamma"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    data_df = hasimg_csv[(hasimg_csv["Detection Date"]>start_date) &
                         (hasimg_csv["Detection Date"]<end_date) &
                         (hasimg_csv["Lab Status"]!=2)
                         ][
        ['Longitude', 'Latitude', "Lab Status"]]

    cv = StratifiedKFold(n_splits=n_splits)
    mean_aucs = np.zeros(len(gammas))
    var_aucs = np.zeros(len(gammas))
    interp_fpr = np.linspace(0, 1, 100)
    for i, gamma in enumerate(gammas):
        # 画平均ROC曲线的两个参数
        X = np.array(data_df[['Longitude', 'Latitude']])
        y = np.array(data_df[['Lab Status']])
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma)
        aucs = np.zeros(n_splits)
        mean_interp_tpr = np.zeros_like(interp_fpr)
        for j, (train, test) in enumerate(cv.split(X, y)):  # 利用模型划分数据集和目标变量 为一一对应的下标
            probas_ = clf.fit(X[train]).decision_function(X[test])  # 训练模型后预测每条样本得到两种结果的概率
            fpr, tpr, thresholds = roc_curve(y[test], probas_)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

            interp_tpr = np.interp(interp_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
            interp_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
            interp_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
            mean_interp_tpr += interp_tpr/n_splits
            aucs[j] = auc(interp_fpr, interp_tpr)
        mean_aucs[i] = aucs.mean()
        var_aucs[i] = aucs.var()

        axs[0].plot(interp_fpr, mean_interp_tpr, '--', label='$\gamma={0}$ AUC = {1:.2f}'.format(gamma, mean_aucs[i]), lw=2)
    axs[0].set_ylim([-0.05, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    axs[0].set_title('Receiver Operating Characteristic Curve')
    axs[0].legend(loc="lower right")

    (_, caps, _) = axs[1].errorbar(gammas, mean_aucs, yerr=var_aucs, uplims=True, lolims=True, fmt='o-', markersize=4)

    axs[1].set_xlabel('$\gamma$')
    axs[1].set_ylabel('Mean Area Under Curve')  # 可以使用中文，但需要导入一些库即字体
    axs[1].set_title("AUC's mean value and variance in {} fold cross validation".format(n_splits))
    plt.tight_layout()
    if save:
        pdf = PdfPages("./img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()
if __name__ == '__main__':
    hasimg_csv = get_hasimg_csv()
    #plot_report_kdes(hasimg_csv)
    plot_mean_roc_auc_by_gamma(hasimg_csv)
    #plot_report_kdes(hasimg_csv)
