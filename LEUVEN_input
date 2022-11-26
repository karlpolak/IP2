from pickle import load 
import numpy as np
import os
from dyntapy.assignments import StaticAssignment
import warnings
warnings.filterwarnings('ignore') # hide warnings

# INPUT FROM HEEDS: TOLL VALUE?  
toll_value = 0
# LINK(S) THAT WE WOULD LIKE TO TOLL
toll_ids = [1490]

# FILL THIS IN ACCORDING TO YOUR NEEDS
city = 'LEUVEN'
buffer_N = 2
# buffer_transit = "45"

# SPECIFY THE PATHS 
buffer = str(buffer_N)
HERE = os.path.dirname(os.path.realpath("__file__"))
data_path = HERE + os.path.sep +'HEEDS_prep'
network_path = data_path + os.path.sep + 'network_centroids_data' + os.path.sep + city + "_" + buffer + '_centroids'
graph_path = HERE + os.path.sep +'HEEDS_prep' + os.path.sep + "od_graph_data" + os.path.sep + city + "_ext_tr_" + buffer + "_9_10"

# LOAD NETWORK FILE
with open(network_path, 'rb') as network_file:
    g = load(network_file)
    print(f'network loaded from f{network_path}')

# LOAD OD-GRAPH
with open(graph_path, 'rb') as f:
    od_graph = load(f)
    print(f'od_graph loaded from f{graph_path}')

# COMPUTE TOLL STRUCTURE BASED ON INPUT
tolls = np.zeros(g.number_of_edges())
for elem in toll_ids:
    tolls[elem] = toll_value

# DO ASSIGNMENT 
assignment = StaticAssignment(g,od_graph, tolls)
result = assignment.run('dial_b')

# OBJECTIVE VALUE = OUTPUT
veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)