from pickle import load 
import sys
sys.path.append("../../..")
from dyntapy.assignments import StaticAssignment
from dyntapy.toll import create_toll_object
import warnings
warnings.filterwarnings('ignore') # hide warnings

# INPUT FROM HEEDS: TOLL VALUE?  
toll_value = 0
# LINK(S) THAT WE WOULD LIKE TO TOLL
toll_ids = [1490]
# TOLL METHOD: single link/zone/cordon
toll_method = 'single'

# FILL THIS IN ACCORDING TO YOUR NEEDS
city = 'BRUSSEL'
buffer = 40

# SPECIFY THE HARDCODED PATHS 
network_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/network_with_centroids/inelastic_BRUSSEL_40"
graph_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/od_graph/inelastic_BRUSSEL_40"

# LOAD NETWORK FILE
with open(network_path, 'rb') as network_file:
    g = load(network_file)
    print(f'network loaded from f{network_path}')

# LOAD OD-GRAPH
with open(graph_path, 'rb') as f:
    od_graph = load(f)
    print(f'od_graph loaded from f{graph_path}')

# CREATE TOLL OBJECT
toll_object = create_toll_object(g, toll_method, toll_ids, toll_value)

# DO ASSIGNMENT 
assignment = StaticAssignment(g,od_graph, toll_object)
result = assignment.run('dial_b')

# OBJECTIVE VALUE = OUTPUT
veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)