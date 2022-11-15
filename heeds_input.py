from pickle import load 
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from dyntapy.demand_data import od_graph_from_matrix
from pyproj import Proj, transform
from dyntapy.assignments import StaticAssignment
import warnings
warnings.filterwarnings('ignore') # hide warnings

# INPUT FROM HEEDS 
toll_value = 0

# Link we decided to toll
toll_id = 1490

# FILL THIS IN ACCORDING TO YOUR NEEDS
city = 'BRUSSEL'
buffer_N = 3
# buffer_transit = "45"

# SPECIFY THE PATHS 
buffer = str(buffer_N)
HERE = os.path.dirname(os.path.realpath("__file__"))
data_path = HERE + os.path.sep +'STA_prep'
zoning_path = data_path + os.path.sep + "shapefile_data" + os.path.sep + city + "_" + buffer+ "_10" + os.path.sep + city + "_" + buffer+ "_10.shp"
network_path = data_path + os.path.sep + 'network_data' + os.path.sep + city + "_" + buffer + '_centroids'
od_path = data_path + os.path.sep + "od_matrix_data_ext_tr" + os.path.sep + city + "_ext_tr_" + buffer + "_9_10.xlsx"

# LOAD NETWORK FILE
with open(network_path, 'rb') as network_file:
    g = load(network_file)
    print(f'network loaded from f{network_path}')

# LOAD OD-MATRIX
od = pd.read_excel(od_path)
od_array = od.to_numpy() # The OD matrix is now stored in a numpy array

# LOAD THE CENTROIDS
zoning = gpd.read_file(zoning_path) 
x_lamb, y_lamb = zoning["X_LAMB"], zoning["Y_LAMB"]
x_lamb, y_lamb = x_lamb.to_numpy(), y_lamb.to_numpy()
inProj, outProj = Proj(init='epsg:31370'), Proj(init='epsg:4326')
x_centroids, y_centroids = transform(inProj,outProj,x_lamb,y_lamb)

# CREATE OD-GRAPH
od_graph = od_graph_from_matrix(od_array,x_centroids,y_centroids)

# DO ASSIGNMENT 
tolls = np.zeros(g.number_of_edges())
tolls[toll_id] = toll_value
assignment = StaticAssignment(g,od_graph, tolls)
methods = ['dial_b']
for method in methods:
    result = assignment.run(method)
    show_network(g, flows = result.flows, notebook=True, show_nodes=False)
    print(f'{method=} ran successfully')
print(result.__dict__.keys())

# OBJECTIVE VALUE = OUTPUT
veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)
print(objective)