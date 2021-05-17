#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
from geopy.distance import geodesic, great_circle


from utils import load_model
from problems import CVRP



# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    
    # route is one sequence, separating different routes with 0 (depot)
    routes = [r[r!=0] for r in np.split(route.cpu().numpy(), np.where(route==0)[0]) if (r != 0).any()]
    depot = data['depot'].cpu().numpy()
    locs = data['loc'].cpu().numpy()
    demands = data['demand'].cpu().numpy() * demand_scale
    capacity = demand_scale # Capacity is always 1
    
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    
    legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        
        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        #assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += great_circle((y, x), (y_prev, x_prev)).km
            # dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d

        dist += great_circle((y_dep, x_dep), (y_prev, x_prev)).km
        # dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                # int(total_route_demand) if round_demand else total_route_demand,
                # int(capacity) if round_demand else capacity,
                dist
            )
        )
        
        qvs.append(qv)
        
    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    ax1.legend(handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)
    return len(routes), total_dist

def test_custom_data(filename, savename=None, demand=.1):
    assert os.path.splitext(filename)[-1] == '.xlsx'
    df = pd.read_excel(filename)
    normalization = MinMaxScaler()
    dfDescribe = df.describe().loc[["min", "max"], ["longitude", "latitude"]]
    dfDescribe = dfDescribe.T
    dfDescribe["span"] = dfDescribe["max"] - dfDescribe["min"]
    df = normalization.fit_transform(df[["longitude", "latitude"]])
    model, _ = load_model('pretrained/cvrp_50/')
    torch.manual_seed(1234)

    dataset = CVRP.make_custom_dataset(df, demand=demand)
    # Need a dataloader to batch instances
    dataloader = DataLoader(dataset, batch_size=1000)

    # Make var works for dicts
    batch = next(iter(dataloader))

    # Run the model
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, tours = model(batch, return_pi=True)
    print([0] + list(i.item() for i in tours[0].data) + [0])

    # [[0, 22, 28, 24, 23, 21, 0],
    # [0, 17, 29, 26, 25, 18, 19, 20, 1, 0],
    # [0, 9, 7, 6, 11, 14, 15, 8, 12, 2, 0],
    # [0, 10, 16, 27, 13, 5, 3, 4, 0]]
    # Plot the results
    for i, (data, tour) in enumerate(zip(dataset, tours)):
        fig, ax = plt.subplots(figsize=(10, 10))
        data["loc"][:, 0] = dfDescribe.loc["longitude", "span"]*data["loc"][:, 0] + dfDescribe.loc["longitude", "min"]
        data["loc"][:, 1] = dfDescribe.loc["latitude", "span"]*data["loc"][:, 1] + dfDescribe.loc["latitude", "min"]
        data["depot"][0] = dfDescribe.loc["longitude", "span"]*data["depot"][0] + dfDescribe.loc["longitude", "min"]
        data["depot"][1] = dfDescribe.loc["latitude", "span"]*data["depot"][1] + dfDescribe.loc["latitude", "min"]
        numV, length = plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)

    if savename is not None:
        pdf = PdfPages("images//cvrp_"+savename+".pdf")
        pdf.savefig()
        pdf.close()
    # else:
    #     plt.show()
    plt.close()
    return numV, length

def plot_(x, y1, y2, savename=None):
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, 'r', label="right");
    ax1.set_ylabel('充电车数量');
    ax1.set_yticks([3,4,5])
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'g', label="left")
    ax2.set_ylabel('总路程');
    ax2.set_xlabel('delta');
    ax2.set_ylim([0, 17]);

    if savename is not None:
        pdf = PdfPages("images//"+savename+".pdf")
        pdf.savefig()
        pdf.close()

if __name__ == '__main__':
    x = np.arange(0.06, 0.16, 0.005)
    ncs = []
    lens = []
    for d in x:
        nc, length = test_custom_data("../data/data.xlsx", demand=d, savename="result03_{:.2f}".format(d))
        ncs.append(nc)
        lens.append(length)
    plot_(x, ncs, lens, "sensitivity")


