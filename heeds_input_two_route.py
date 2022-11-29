import warnings
warnings.filterwarnings('ignore') # hide warnings
import numpy as np
from pickle import load
from dyntapy.assignments import StaticAssignment

# INPUT FROM HEEDS: TOLL VALUE?  
toll_value = 0
# LINK(S) THAT WE WOULD LIKE TO TOLL
toll_ids = [1490]

network_path = "C:/Users/anton/IP2/HEEDS_prep/network_centroids_data/two_route"
graph_path = "C:/Users/anton/IP2/HEEDS_prep/od_graph_data/two_route"
with open(network_path, 'rb') as network_file:
    network = load(network_file)
    print(f'network saved at f{network_path}')
with open(graph_path, 'rb') as f:
    od_graph = load(f)
    print(f'od_graph saved at f{graph_path}')


tolls = np.zeros(g.number_of_edges())
for elem in toll_ids:
    tolls[elem] = toll_value
assignment = StaticAssignment(network, od_graph, tolls) # TODO fix tolls
result = assignment.run('dial_b')

veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)