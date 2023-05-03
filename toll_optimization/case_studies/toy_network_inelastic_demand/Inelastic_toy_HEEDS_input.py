import warnings
warnings.filterwarnings('ignore') # hide warnings
import numpy as np
from pickle import load
from dyntapy.assignments import StaticAssignment
from dyntapy.toll import create_toll_object

# INPUT FROM HEEDS = toll value(s)
toll_value = 0
# OUR INPUT: toll scheme and links to toll
toll_ids = [2]
toll_method = 'single'

# Retrieve network and od-graph
network_path = "C:/Users/anton/IP2/HEEDS_prep/network_centroids_data/two_route"
graph_path = "C:/Users/anton/IP2/HEEDS_prep/od_graph_data/two_route"
with open(network_path, 'rb') as network_file:
    g = load(network_file)
    print(f'network saved at f{network_path}')
with open(graph_path, 'rb') as f:
    od_graph = load(f)
    print(f'od_graph saved at f{graph_path}')

# Create toll object
toll_object = create_toll_object(g, toll_method, toll_ids, toll_value)
assignment = StaticAssignment(g, od_graph, toll_object)
result = assignment.run('dial_b')

# Output for HEEDS
veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)