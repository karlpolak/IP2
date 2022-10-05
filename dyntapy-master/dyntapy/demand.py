#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
#
from collections import OrderedDict
from warnings import warn
import networkx as nx
import numpy as np
from numba import njit, float32
from numba.core.types import ListType, uint32
from numba.typed.typedlist import List
from numba.experimental import jitclass

from dyntapy.csr import F32CSRMatrix, UI32CSRMatrix, csr_prep, f32csr_type
from dyntapy.supply import Network


class DynamicDemand:
    """

    Parameters
    ----------
    od_graphs :list of networkx.DiGraph
    insertion_times: numpy.ndarray
        times for the demand to be loaded into the network

    Notes
    -----

    The insertion times need to be within the bounds of the defined simulation time
    that is passed to the DynamicAssignment

    See Also
    --------

    dyntapy.assignments.DynamicAssignment

    dyntapy.demand.time.SimulationTime
    """

    def __init__(self, od_graphs, insertion_times):
        if type(od_graphs) is not list:
            raise ValueError
        for item in od_graphs:
            if type(item) is not nx.DiGraph:
                raise ValueError
        self.od_graphs = od_graphs
        self.insertion_times = np.array(insertion_times)

    def _get_sparse_repr(self, time):
        """
        Parameters
        ----------
        time : float or integer, time slice to be retrieved

        Returns
        -------
        scipy.lil_matrix of trip table for given time slice
        """
        graph = self._get_od_graph(time)
        return nx.to_scipy_sparse_matrix(graph, weight="flow", format="lil")

    def _get_od_graph(self, time):
        _id = np.argwhere(self.insertion_times == time)[0][0]
        graph: nx.DiGraph = self.od_graphs[_id]
        return graph


spec_demand = OrderedDict(
    [
        ("to_destinations", f32csr_type),
        ("to_origins", f32csr_type),
        ("origins", uint32[:]),
        ("destinations", uint32[:]),
        ("time_step", uint32),
    ]
)


def build_internal_static_demand(od_graph: nx.DiGraph):
    """

    builds InternalStaticDemand

    Parameters
    ----------
    od_graph: networkx.DiGraph

    See Also
    --------

    dyntapy.demand.InternalStaticDemand

    """
    try:
        # available since networkx version 2.7.0, February 2022
        lil_demand = nx.to_scipy_sparse_array(od_graph, weight="flow", format="lil")
    except AttributeError:
        # will be deprecated for networkx 3.0
        lil_demand = nx.to_scipy_sparse_matrix(od_graph, weight="flow", format="lil")
    tot_centroids = od_graph.number_of_nodes()
    row = np.asarray(lil_demand.nonzero()[0])
    col = np.asarray(lil_demand.nonzero()[1])
    vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)

    intra_zonal = []
    for idx, (r, c, v) in enumerate(zip(row, col, vals)):
        if r == c:
            intra_zonal.append(idx)
    if len(intra_zonal) > 0:
        warn("intra-zonal traffic is ignored")
        # intra-zonal traffic, ignored..
    row = np.delete(row, intra_zonal)
    vals = np.delete(vals, intra_zonal)
    col = np.delete(col, intra_zonal)
    index_array_to_d = np.column_stack((row, col))
    index_array_to_o = np.column_stack((col, row))
    to_destinations = F32CSRMatrix(
        *csr_prep(index_array_to_d, vals, (tot_centroids, tot_centroids))
    )
    to_origins = F32CSRMatrix(
        *csr_prep(index_array_to_o, vals, (tot_centroids, tot_centroids))
    )
    return InternalStaticDemand(
        to_origins,
        to_destinations,
        to_destinations.get_nnz_rows(),
        to_origins.get_nnz_rows(),
        np.uint32(0),
    )


@jitclass(spec_demand)
class InternalStaticDemand(object):
    """
    internal specification of static demand

    Parameters
    ----------
    to_origins: F32CSRMatrix
        sparse OD matrix destinations to origins
    to_destinations: F32CSRMatrix
        sparse OD matrix destinations to origins
    origins: numpy.ndarray
        all origins with non-zero flow
    destinations: numpy.ndarray
        all destinations with non-zero flow
    time_step: int


    See Also
    --------

    should be initiated with

    dyntapy.demand.build_internal_static_demand

    """

    def __init__(
        self,
        to_origins: F32CSRMatrix,
        to_destinations: F32CSRMatrix,
        origins,
        destinations,
        time_step,
    ):
        self.to_destinations = to_destinations  # csr matrix origins x destinations
        self.to_origins = to_origins  # csr destinations x origins
        self.origins = origins  # array of active origin id's
        self.destinations = destinations  # array of active destination id's
        self.time_step = time_step  # time at which this demand is added to the network


try:
    spec_simulation = [
        ("next", InternalStaticDemand.class_type.instance_type),
        ("demands", ListType(InternalStaticDemand.class_type.instance_type)),
        ("__time_step", uint32),
        ("tot_time_steps", uint32),
        ("all_active_destinations", uint32[:]),
        ("all_active_destination_links", uint32[:]),
        ("all_active_origins", uint32[:]),
        ("all_centroids", uint32[:]),
        ("tot_centroids", uint32),
        ("tot_active_destinations", uint32),
        ("tot_active_origins", uint32),
        ("loading_time_steps", uint32[:]),
    ]
except Exception:
    # numba disabled
    spec_simulation = None


@jitclass(spec_simulation)
class InternalDynamicDemand(object):
    """

    internal specification of dynamic demand

    should be initialized with build_dynamic_demand function

    Parameters
    ----------
    demands: list of InternalStaticDemand
    tot_time_steps: int
    tot_centroids: int
    in_links: UI32CSRMatrix
        in_links for all nodes in the networks

    See Also
    --------

    dyntapy.demand.build_internal_dynamic_demand

    """

    def __init__(self, demands, tot_time_steps, tot_centroids, in_links: UI32CSRMatrix):
        self.demands = demands
        self.next = demands[0]
        self.loading_time_steps = _get_loading_time_steps(demands)
        # time step traffic is loaded into the network
        self.all_active_destinations = _get_all_destinations(demands)
        self.all_active_destination_links = _get_destination_links(
            self.all_active_destinations, in_links
        )
        self.all_active_origins = _get_all_origins(demands)
        self.tot_active_origins = self.all_active_origins.size
        self.tot_active_destinations = self.all_active_destinations.size
        self.all_centroids = np.arange(
            tot_centroids, dtype=np.uint32
        )  # for destination/origin based labels
        self.tot_time_steps = np.uint32(tot_time_steps)
        self.tot_centroids = np.uint32(tot_centroids)

    def _is_loading(self, t):
        _ = np.argwhere(self.loading_time_steps == t)
        if _.size == 1:
            return True
        elif _.size > 1:
            raise Exception(
                "ValueError, multiple StaticDemand objects with identical time label"
            )
        else:
            return False

    def _get_demand(self, t):
        _id = np.argwhere(self.loading_time_steps == t)[0][0]
        return self.demands[_id]


@njit(cache=True)
def _get_loading_time_steps(demands):
    loading = np.empty(len(demands), dtype=np.uint32)
    for _id, demand in enumerate(demands):
        demand: InternalStaticDemand
        t = demand.time_step
        loading[_id] = np.uint32(t)
    return loading


@njit
def _get_all_destinations(demands):
    if len(demands) < 1:
        raise AssertionError
    previous = demands[0].destinations
    if len(demands) == 1:
        return previous
    for demand in demands[1:]:
        demand: InternalStaticDemand
        current = np.concatenate((demand.destinations, previous))
        previous = current
    return np.unique(current)


@njit
def _get_all_origins(demands):
    if len(demands) < 1:
        raise AssertionError
    previous = demands[0].origins
    if len(demands) == 1:
        return previous
    for demand in demands[1:]:
        demand: InternalStaticDemand
        current = np.concatenate((demand.origins, previous))
        previous = current
    return np.unique(current)


@njit(cache=True)
def _get_destination_links(destinations: np.ndarray, in_links: UI32CSRMatrix):
    """
    Parameters
    ----------
    destinations : destinations to get links for
    in_links : CSRMatrix, in_links for all nodes in the network.
     Assumes that every centroid only has one in_link.

    Returns
    -------
    array containing the corresponding connector for each destination

    """
    destinations_link = np.empty(destinations.size, dtype=np.uint32)
    for d_id, destination in enumerate(destinations):
        assert in_links.get_nnz(destination).size == 1
        for link in in_links.get_nnz(destination):
            destinations_link[d_id] = link
    return destinations_link


spec_time = [
    ("start", float32),
    ("end", float32),
    ("step_size", float32),
    ("tot_time_steps", uint32),
]


@jitclass(spec_time)
class SimulationTime(object):
    """
    specification of time discretization, units are always assumed in hours

    Parameters
    ----------
    start : int
    end : int
    step_size : float

    """

    def __init__(self, start, end, step_size):
        # TODO: start always 0
        self.start = start
        self.end = end
        self.step_size = step_size
        self.tot_time_steps = np.uint32(np.ceil((end - start) / step_size))


def build_internal_dynamic_demand(
    dynamic_demand: DynamicDemand, simulation_time: SimulationTime, network: Network
):
    """
    instantiates InternalDynamicDemand

    Parameters
    ----------
    dynamic_demand: DynamicDemand
    simulation_time: SimulationTime
    network: Network

    Returns
    -------

    """
    # finding closest time step for defined demand insertion times
    # each time is translated to an index and element of [0,1, ..., tot_time_steps]
    insertion_times = np.array(
        [
            np.argmin(
                np.abs(
                    np.arange(simulation_time.tot_time_steps)
                    * simulation_time.step_size
                    - time
                )
            )
            for time in dynamic_demand.insertion_times
        ],
        dtype=np.uint32,
    )
    demand_data = [
        dynamic_demand._get_sparse_repr(t) for t in dynamic_demand.insertion_times
    ]

    if not np.all(
        dynamic_demand.insertion_times[1:] - dynamic_demand.insertion_times[:-1] >=
        simulation_time.step_size
    ):
        raise ValueError(
            "insertion times are assumed to be monotonously increasing."
            " The minimum difference between "
            "two "
            "insertions is the internal simulation time step"
        )
    if max(dynamic_demand.insertion_times > 24):
        raise ValueError("internally time is restricted to 24 hours")

    static_demands = List()
    rows = [
        np.asarray(lil_demand.nonzero()[0], dtype=np.uint32)
        for lil_demand in demand_data
    ]
    cols = [
        np.asarray(lil_demand.nonzero()[1], dtype=np.uint32)
        for lil_demand in demand_data
    ]
    tot_centroids = np.uint32(
        max([trip_graph.number_of_nodes() for trip_graph in dynamic_demand.od_graphs])
    )
    for internal_time, lil_demand, row, col in zip(
        insertion_times, demand_data, rows, cols
    ):
        vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
        index_array_to_d = np.column_stack((row, col))
        index_array_to_o = np.column_stack((col, row))
        to_destinations = F32CSRMatrix(
            *csr_prep(index_array_to_d, vals, (tot_centroids, tot_centroids))
        )
        to_origins = F32CSRMatrix(
            *csr_prep(index_array_to_o, vals, (tot_centroids, tot_centroids))
        )
        static_demands.append(
            InternalStaticDemand(
                to_origins,
                to_destinations,
                to_destinations.get_nnz_rows(),
                to_origins.get_nnz_rows(),
                np.uint32(internal_time),
            )
        )
    return InternalDynamicDemand(
        static_demands,
        simulation_time.tot_time_steps,
        tot_centroids,
        network.nodes.in_links,
    )
