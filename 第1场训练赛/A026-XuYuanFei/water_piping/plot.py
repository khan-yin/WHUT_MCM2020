import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def plot_mat(mat, save=True, name="mat"):
    plt.figure()
    plt.imshow(mat, cmap=plt.cm.hot)
    plt.colorbar()
    if save:
        pdf = PdfPages("img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_map(station, save=True, pointSize=25, name="map"):
    # print(station["classID"])
    plt.figure()
    plt.scatter(station["X"], station["Y"], s=pointSize-10*station["classID"], c=station["classID"], cmap="Set1")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    if save:
        pdf = PdfPages("img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_pipeline(station, edges1, edges2, upgradeStationID=None, save=True, pointSize=50, name="pipeline"):
    plt.figure()
    for edge in edges2:
        x0, x1 = station["X"][edge[0]], station["X"][edge[1]]
        y0, y1 = station["Y"][edge[0]], station["Y"][edge[1]]
        plt.plot([x0, x1], [y0, y1], c="darkorange")
    for edge in edges1:
        x0, x1 = station["X"][edge[0]], station["X"][edge[1]]
        y0, y1 = station["Y"][edge[0]], station["Y"][edge[1]]
        plt.plot([x0, x1], [y0, y1], c="r")
    plt.scatter(station.loc[station["classID"] == 0]["X"],
                station.loc[station["classID"] == 0]["Y"],
                s=pointSize, marker="*", c="r")
    plt.scatter(station.loc[station["classID"] == 1]["X"],
                station.loc[station["classID"] == 1]["Y"],
                s=pointSize/2, marker="s", c="b")
    plt.scatter(station.loc[station["classID"] == 2]["X"],
                station.loc[station["classID"] == 2]["Y"],
                s=pointSize/4, c="dimgray")
    if upgradeStationID is not None:
        plt.scatter(station["X"][upgradeStationID],
                    station["Y"][upgradeStationID],
                    s=2*pointSize, marker="+", c="g")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    if save:
        pdf = PdfPages("img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_tree(station, edges1, tree, selectedAvailable=None, candidateOverload=None,
              upgradeStationID=None, extraEdges=None, cutEdge=None,
              annotateIIStation=True, save=True, pointSize=50, name="tree"):
    plt.figure()
    for edge in edges1:
        x0, x1 = station["X"][edge[0]], station["X"][edge[1]]
        y0, y1 = station["Y"][edge[0]], station["Y"][edge[1]]
        plt.plot([x0, x1], [y0, y1], c="r")
    for stationID, nextSet in tree.tree.items():
        for next in nextSet:
            if (stationID, next) == cutEdge:
                linestyle = (0, (1, 1))
                c = "g"
            else:
                linestyle = 'solid'
                c = "darkorange"
            x0, x1 = station["X"][stationID], station["X"][next]
            y0, y1 = station["Y"][stationID], station["Y"][next]
            plt.plot([x0, x1], [y0, y1], c=c, linestyle=linestyle)

    plt.scatter(station.loc[station["classID"] == 0]["X"],
                station.loc[station["classID"] == 0]["Y"],
                s=pointSize, marker="*", c="r")
    plt.scatter(station.loc[station["classID"] == 1]["X"],
                station.loc[station["classID"] == 1]["Y"],
                s=pointSize/2, marker="s", c="b")
    plt.scatter(station.loc[station["classID"] == 2]["X"],
                station.loc[station["classID"] == 2]["Y"],
                s=pointSize/4, c="dimgray")
    IStationIDLst = station.loc[station["classID"] == 1]["id"]
    for IStationID in IStationIDLst:
        plt.annotate('{:.1f}'.format(tree.length_from_station[IStationID]),
                     (station["X"][IStationID], station["Y"][IStationID]), fontsize=10
                     )
    if annotateIIStation:
        IIStationIDLst = station.loc[station["classID"] == 2]["id"]
        for IIStationID in IIStationIDLst:
            plt.annotate('{:.1f}'.format(tree.length_from_station[IIStationID]),
                         (station["X"][IIStationID], station["Y"][IIStationID]), fontsize=6
                         )
    if selectedAvailable is not None:
        for stationID in selectedAvailable:
            plt.scatter(station["X"][stationID],
                        station["Y"][stationID],
                        s=1.5*pointSize, c='', marker="o", edgecolors="g")
    if candidateOverload is not None:
        for stationID in candidateOverload:
            plt.scatter(station["X"][stationID],
                        station["Y"][stationID],
                        s=1.5*pointSize, c='', marker="o", edgecolors="r")
    if extraEdges is not None:
        for extraEdge in extraEdges:
            x0, x1 = station["X"][extraEdge[0]], station["X"][extraEdge[1]]
            y0, y1 = station["Y"][extraEdge[0]], station["Y"][extraEdge[1]]
            plt.plot([x0, x1], [y0, y1], c="g", linestyle=(0, (1, 1)))
    if upgradeStationID is not None:
        plt.scatter(station["X"][upgradeStationID],
                    station["Y"][upgradeStationID],
                    s=1.5*pointSize, marker="+", c="g")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    if save:
        pdf = PdfPages("img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()