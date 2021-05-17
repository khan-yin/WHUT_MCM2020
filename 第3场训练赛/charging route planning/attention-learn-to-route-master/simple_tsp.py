#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import load_model
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from geopy.distance import geodesic, great_circle

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def make_oracle(model, xy, temperature=1.0):
    
    num_nodes = len(xy)
    
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
    
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            #assert np.allclose(p.sum().item(), 1)
        return p.numpy()
    
    return oracle

def calc_length(xs, ys):
    length = 0
    for i in range(len(xs)-1):
        cityi = (ys[i], xs[i])
        cityj = (ys[i+1], xs[i+1])
        length += geodesic(cityi, cityj).km
    cityi = (ys[-1], xs[-1])
    cityj = (ys[0], xs[0])
    length += geodesic(cityi, cityj).km
    return length

def plot_tsp(xy, tour, depot, ax1):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    # d = np.sqrt(dx * dx + dy * dy)
    # lengths = d.cumsum()[-1]
    lengths = calc_length(xs, ys)

    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color='blue')
    # Starting node
    ax1.scatter([depot[0]], [depot[1]], s=100, color='red')
    
    # Arcs
    ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
    )
    
    ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), lengths))

def test_custom_data(filename, savename=None):
    assert os.path.splitext(filename)[-1] == '.xlsx'
    df = pd.read_excel(filename)[["longitude", "latitude"]]
    normalization = MinMaxScaler()
    dfDescribe = df.describe().loc[["min", "max"], ["longitude", "latitude"]]
    dfDescribe = dfDescribe.T
    dfDescribe["span"] = dfDescribe["max"] - dfDescribe["min"]
    dfnormalized = normalization.fit_transform(df)
    xy = np.array(dfnormalized)
    model, _ = load_model('pretrained/tsp_100/')
    model.eval()  # Put in evaluation mode to not track gradients
    oracle = make_oracle(model, xy)

    sample = False
    tour = []
    tour_p = []
    while (len(tour) < len(xy)):
        p = oracle(tour)

        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
            # i = np.random.multinomial(1, p)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)
    # [0, 2, 1, 9, 8, 12, 10, 5, 13, 16, 27, 15, 14, 11, 6,
    # 7, 18, 25, 26, 29, 19, 20, 17, 21, 23, 22, 24, 28, 3, 4]
    print(tour)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsp(np.array(df), tour, np.array(df)[0], ax)
    if savename is not None:
        pdf = PdfPages("images//cvrp_"+savename+".pdf")
        pdf.savefig()
        pdf.close()
    else:
        plt.show()

if __name__ == '__main__':
    test_custom_data("../data/data.xlsx", "result01")


