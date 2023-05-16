import numpy as np
import warnings
warnings.filterwarnings('ignore') # hide warnings
import sys
sys.path.append("../../..")
from pickle import load
from dyntapy.assignments import StaticAssignment
from dyntapy.toll import create_toll_object
from dyntapy.demand_data import od_graph_from_matrix

# ------------------------------------ CHANGE TO YOUR NEEDS AND FILES: START -------------------------------------
# Initializing toy network: retrieving network and creating some demand 
network_path = "C:/Users/anton/IP2/toll_optimization/data_map/HEEDS_input/network_with_centroids/elastic_toy"
old_od = np.zeros(4).reshape((2, 2))
old_od[0, 1] = 10000
centroid_x = np.array([-1, 3])
centroid_y = np.array([2, 2])
old_graph = od_graph_from_matrix(old_od, centroid_x, centroid_y)

# Parameters for elastic demand (same values as in Elastic_toy_STA.ipynb)
B = 0.05
A = 528

# Creating our tolling scheme: links to toll and type of scheme
toll_ids = [2]
toll_method = 'single'
# Heeds chooses the toll value as input 
toll_value = 0
# ------------------------------------ CHANGE TO YOUR NEEDS AND FILES: END -------------------------------------

# First STA iteration to start modifying OD-matrix
with open(network_path, 'rb') as network_file:
    g = load(network_file)
toll_object = create_toll_object(g, toll_method, toll_ids, toll_value)
assignment = StaticAssignment(g, old_graph, toll_object)
result = assignment.run('dial_b')
skims = result.skim
new_od = np.zeros(4).reshape((2, 2))
new_od[0,1] = (A-skims)/B

# Repeat until convergence to final OD-matrix
i = 1 
max_iterations = 3
while abs((np.linalg.norm(new_od) - np.linalg.norm(old_od))/np.linalg.norm(old_od)) > 0.05 and i < max_iterations:
    old_od = new_od
    old_graph =  od_graph_from_matrix(old_od, centroid_x, centroid_y)
    assignment = StaticAssignment(g, old_graph, toll_object)
    result = assignment.run('dial_b')
    skims = result.skim
    new_od = np.zeros(4).reshape((2, 2))
    new_od[0,1] = (A-skims)/B
    i += 1

# Output for HEEDS = objective for STA on final OD
veh_hour_per_link = result.link_costs * result.flows
objective = sum(veh_hour_per_link)