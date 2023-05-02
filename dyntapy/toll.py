import numpy as np
from dyntapy.supply_data import build_network

class Toll:
    # Tolling method can be either 'single', 'cordon' or 'zone'. 
    # 
    # For single, you must provide a single link_id (integer) and single toll value to raise (floating number)
    # 
    # For zone, you must provide the set of link_ids (list of integers) and a single toll value (floating number).
    # The toll value is a toll/kilometer, and the total toll on one link is found by multiplying it with the link length. 
    #
    # For cordon, you must provide the set of link_ids (list of integers) and provide one or more toll values (either floating number or list of floating numbers).
    # Multiple toll values allow setting different toll values on different links. Note that the indices in the list should match (value at index 1 is set as toll for link at index 1, ...)

    def __init__(self, network, method, link_ids, values) -> None:
        self.toll_method = method
        self.values = values
        self.link_ids = link_ids
        self.toll_costs = []
        self.set_toll_costs(network)

    def get_toll_costs(self):
        return self.toll_costs
    
    def set_toll_costs(self, network):
        toll_costs = np.zeros(network.number_of_edges())
        if self.toll_method == 'single':
            toll_costs[self.link_ids] = self.values 
        elif self.toll_method == 'zone':
            internal_network = build_network(network)
            for id in self.link_ids:
                    toll_costs[id] = self.values * internal_network.links.length[id]
        else:  
            if type(self.values) == int:
                for id in self.link_ids:
                    toll_costs[id] = self.values
            else: 
                for i in range(0,len(self.values)):
                    toll_costs[self.link_ids[i]] = self.values[i]
        
        self.toll_costs = toll_costs

def create_toll_object(g, toll_method, toll_id, toll_value):
    return Toll(g, toll_method, toll_id, toll_value)

