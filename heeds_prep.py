# This file provides the network file with centroids and the od_graph. 
# This way of working is useful because we only need to compute those files once, 
# instead of in each iteration made by HEEDS. 
import warnings
import os
import geopandas as gpd
import pandas as pd
from dyntapy.supply_data import road_network_from_place, relabel_graph
from dyntapy.demand_data import add_centroids, od_graph_from_matrix
from pyproj import Proj, transform
from pickle import dump
warnings.filterwarnings('ignore') # hide warnings

# FILL IN THOSE VARIABLES ACCORDING TO YOUR NEEDS
city = 'BRUSSEL'
buffer_N = 3
#buffer_transit = "45"  # Only used in the name of the file we save the network to. 


# Specify all data paths 
buffer = str(buffer_N) 
HERE = os.path.dirname(os.path.realpath("__file__"))
STA_data_path = HERE + os.path.sep +'STA_prep'
HEEDS_data_path = HERE + os.path.sep +'HEEDS_prep'
zoning_path = STA_data_path + os.path.sep + 'shapefile_data' + os.path.sep + city + "_" + buffer + '_10' + os.path.sep + city + "_" + buffer + '_10.shp'
network_path = HEEDS_data_path + os.path.sep + 'network_centroids_data' + os.path.sep + city + "_" + buffer + '_centroids'
od_path = STA_data_path + os.path.sep + "od_matrix_data_ext_tr" + os.path.sep + city + "_ext_tr_" + buffer + "_9_10.xlsx"
graph_path = HERE + os.path.sep +'HEEDS_prep' + os.path.sep + "od_graph_data" + os.path.sep + city + "_ext_tr_" + buffer + "_9_10"


# Create network file, add centroids to it and store the result. 
buffer_N = buffer_N * 1000
g = road_network_from_place(city, buffer_dist_close=buffer_N)
g = relabel_graph(g)
zoning = gpd.read_file(zoning_path) 
x_lamb, y_lamb = zoning["X_LAMB"], zoning["Y_LAMB"]
x_lamb, y_lamb = x_lamb.to_numpy(), y_lamb.to_numpy()
inProj, outProj = Proj(init='epsg:31370'), Proj(init='epsg:4326')
x_centroids, y_centroids = transform(inProj, outProj, x_lamb, y_lamb)
g = add_centroids(g, x_centroids,y_centroids,k=1, method='link')
g = relabel_graph(g)
with open(network_path, 'wb') as network_file:
    dump(g, network_file)
    print(f'network saved at f{network_path}')


# Create od_graph and store the result. 
od = pd.read_excel(od_path)
od_array = od.to_numpy() # The OD matrix is now stored in a numpy array
od_graph = od_graph_from_matrix(od_array, x_centroids, y_centroids) 
with open(graph_path, 'wb') as f:
    dump(od_graph,f)
    print(f'od_graph saved at f{od_path}')

