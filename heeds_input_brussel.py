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
city = 'BRUSSEL'
buffer_N = 3
# buffer_transit = "45"

# SPECIFY THE HARDCODED PATHS 
network_path = "C:/Users/anton/IP2/HEEDS_prep/network_centroids_data/BRUSSEL_3_centroids"
graph_path = "C:/Users/anton/IP2/HEEDS_prep/od_graph_data/BRUSSEL_ext_tr_3_9_10"

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