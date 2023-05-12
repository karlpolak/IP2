import sys
sys.path.append("../../..")
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from dyntapy.assignments import StaticAssignment
from dyntapy.toll import create_toll_object
from dyntapy.demand_data import od_graph_from_matrix
from pickle import load 
import warnings
warnings.filterwarnings('ignore') # hide warnings

HERE = os.path.dirname(os.path.realpath("__file__"))
data_path = HERE + os.path.sep + os.pardir + os.path.sep + os.pardir + os.path.sep + 'data_map'
links_crossing_cordon = gpd.read_file(data_path  + os.path.sep + 'QGIS' + os.path.sep + "links_crossing_cordon.shp")
links_in_cordon = gpd.read_file(data_path  + os.path.sep + 'QGIS' + os.path.sep + "links_in_cordon.shp")

# 4 PARAMETERS YOU SHOULD FILL IN YOURSELF, DEPENDING ON YOUR NEEDS
# 1) CITY OF CASE STUDY
city = 'BRUSSEL'
# 2) CONSIDERED RADIUS/BUFFER OF NETWORK 
buffer = 40
# 3) LINK IDS OF NETWORK THAT YOU WOULD LIKE TO TOLL 
# TODO nog testen, maar zou moeten kloppen
toll_ids = links_crossing_cordon['link_id']
# 4) TOLLING METHOD THAT YOU WOULD LIKE TO SET: single/cordon/zone
toll_method = 'cordon'

# INPUT FROM HEEDS: TOLL VALUE
toll_value = 0

# Load network, original/starting OD, A_matrix and B_matrix 
network_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/network_with_centroids/inelastic_BRUSSEL_40"
od_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/od_graph/elastic_BRUSSEL_40.xlsx"
A_matrix_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/elastic/A_matrix_BRUSSEL_40"
B_matrix_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/elastic/B_matrix_BRUSSEL_40"
x_centroids_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/elastic/x_centroids_BRUSSEL_40"
y_centroids_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/elastic/y_centroids_BRUSSEL_40"

with open(network_path, 'rb') as network_file:
    g = load(network_file)
with open(A_matrix_path, 'rb') as matrix:
    A = load(matrix)
with open(B_matrix_path, 'rb') as matrix:
    B = load(matrix)
original_od = pd.read_excel(od_path)
original_od = original_od.to_numpy()
x_centroids = np.loadtxt(x_centroids_path)
y_centroids = np.loadtxt(y_centroids_path)
original_od_graph = od_graph_from_matrix(original_od,x_centroids,y_centroids) 


# First assignment with toll to achieve new OD 
toll_object = create_toll_object(g, toll_method, toll_ids, toll_value)
assignment = StaticAssignment(g, original_od_graph, toll_object)
result = assignment.run('dial_b')
new_od = (A-result.skim)/B

# Make sure zero rows are not actually zero, by setting at least a flow of 1 for each O or D. 
indices_zero_rows = np.where((new_od==0).all(axis=1))[0]
indices_zero_cols = np.where((new_od==0).all(axis=0))[0]
for i in indices_zero_rows:
    j = np.random.choice(new_od.shape[1])
    new_od[i,j] = 1
for j in indices_zero_cols:
    i = np.random.choice(new_od.shape[1])
    new_od[i,j] = 1

# Loop over STAs until stopping criterion met or maximum iterations
i = 1 
max_iterations = 3

while abs((np.linalg.norm(new_od) - np.linalg.norm(old_od))/np.linalg.norm(old_od)) > 0.05 and i < max_iterations:
    print("started iteration:", i)
    old_od = new_od
    old_graph =  od_graph_from_matrix(old_od, x_centroids, y_centroids)
    assignment = StaticAssignment(g, old_graph, toll_object)
    result = assignment.run('dial_b')
    new_od = old_od + ((A-result.skim)/B - old_od)/i
    indices_zero_rows = np.where((new_od==0).all(axis=1))[0]
    indices_zero_cols = np.where((new_od==0).all(axis=0))[0]
    for i in indices_zero_rows:
        j = np.random.choice(new_od.shape[1])
        new_od[i,j] = 1
    for j in indices_zero_cols:
        i = np.random.choice(new_od.shape[1])
        new_od[i,j] = 1
    i += 1

# Compute objective value of STA with converged elastic OD matrix. 
veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)