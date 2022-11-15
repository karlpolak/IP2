from pickle import load 
import os
from dyntapy.visualization import show_network


# FILL THIS IN ACCORDING TO YOUR NEEDS
city = 'Brussels'
buffer_N = 1
buffer_transit = "45"
buffer_N_string = str(buffer_N)



HERE = os.path.dirname(os.path.realpath("__file__"))
network_path = HERE + os.path.sep +'STA_prep' + os.path.sep + 'network_data' + os.path.sep + 'BRUSSEL_' + buffer_N_string + '_centroids'

with open(network_path, 'rb') as network_file:
    g = load(network_file)
    print(f'network saved at f{network_path}')

show_network(g)
