# This file is ran before running the heeds script, so that we only have to create the network file once! 
import warnings
import os
import geopandas as gpd
from dyntapy.supply_data import road_network_from_place, relabel_graph
from dyntapy.demand_data import add_centroids
from dyntapy.visualization import show_network
from pyproj import Proj, transform
from pickle import dump
warnings.filterwarnings('ignore') # hide warnings

# FILL IN THOSE VARIABLES ACCORDING TO YOUR NEEDS
city = 'Brussels'
buffer_N = 1
buffer_transit = "45"  # Only used in the name of the file we save the network to. 


# Create network file
buffer_N_string = str(buffer_N) 
buffer_N = buffer_N * 1000
g = road_network_from_place(city, buffer_dist_close=buffer_N)

# Add the centroids to the network, which can be derived from the zoning file. 
HERE = os.path.dirname(os.path.realpath("__file__"))
zoning_path = HERE + os.path.sep +'STA_prep' + os.path.sep + 'shapefile_data' + os.path.sep + 'BRUSSEL_' + buffer_N_string + '_10' + os.path.sep + 'BRUSSEL_' + buffer_N_string + '_10.shp'
network_path = HERE + os.path.sep +'STA_prep' + os.path.sep + 'network_data' + os.path.sep + 'BRUSSEL_' + buffer_N_string + '_centroids'
zoning = gpd.read_file(zoning_path) 
x_lamb = zoning["X_LAMB"]
x_lamb = x_lamb.to_numpy()
y_lamb = zoning["Y_LAMB"]
y_lamb = y_lamb.to_numpy()
inProj = Proj(init='epsg:31370')
outProj = Proj(init='epsg:4326')
x_centroids, y_centroids = transform(inProj,outProj,x_lamb,y_lamb)
g = add_centroids(g, x_centroids,y_centroids,k=1, method='link')
g = relabel_graph(g)
show_network(g)
with open(network_path, 'wb') as network_file:
    dump(g, network_file)
    print(f'network saved at f{network_path}')
