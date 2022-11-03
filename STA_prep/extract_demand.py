import pandas as pd
import geopandas as gpd
import re
import numpy as np
import pathlib
from numba import njit, prange
from scipy.spatial import cKDTree
from warnings import warn
from pyproj import Transformer



def find_mapping(X1: np.ndarray, Y1: np.ndarray, X2: np.ndarray, Y2: np.ndarray):
    """
    maps from zones in zoning 1 defined by centroid locations X1, Y1 to zones in zoning 2 with X2,Y2 minimizing
    straight line distance.
    coordinates should be projected for accurate results on distance.
    Parameters
    ----------

    X1 : centroids x coordinates zoning 1
    Y1 : centroids y coordinates zoning 1
    X2 : centroids x coordinates zoning 2
    Y2 : centroids y coordinates zoning 2

    Returns
    -------
    mapping: array [int]
    such that mapping[i] = k indicates that centroid i with coordinates (X1[i],Y1[i]) is mapped to
    centroid k with (X2[k],Y2[k])
    """
    tree = cKDTree(data=np.vstack([X1, Y1]).T, compact_nodes=True, balanced_tree=True)
    points = np.vstack([X2, Y2]).T
    dist, idx = tree.query(points, k=1)
    # k nearest neighbors, nearest neighbors of points (all centroids) in cKDTree (cordon/internal centroids)
    # dist =distance, idx = id of the nearest neighbors
    return dist, idx


def load_data_mow():
    # loads the zoning and od table information, cleans and transforms them to
    # returns complete od table as full numpy array
    # returns complete zoning as geodataframe with explicit information on centroid location (x, y coordinates) (inferred from polygon)

    here = pathlib.Path(__file__).parent
    # The directory containing this file
    zoning_file = (here / "Zonering" / "Zonering_RVM_VLA.shp")
    dynamic_od_table_csv = (here / "Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.CSV")
    belgium_od_table = pd.read_csv(dynamic_od_table_csv, header=2)
    belgium_od_table.columns = belgium_od_table.columns.str.replace('\'', '')
    tot_time_steps = [int(e) for e in re.split("[^0-9]", belgium_od_table.columns[-1]) if e != ''][0]# ???
    tot_destinations = belgium_od_table['B'].max()
    tot_origins = belgium_od_table['H'].max()
    belgium_od = np.zeros((tot_origins, tot_destinations, tot_time_steps), dtype=np.float32)
    belgium_od_table = belgium_od_table.to_numpy()

    # make copying fast with numba...
    @njit(parallel=True)
    def parse(matrix1, matrix2):
        for row in prange(matrix1.shape[0]):
            i = np.uint32(matrix1[row][0] - 1)
            j = np.uint32(matrix1[row][1] - 1)
            matrix2[i][j] = matrix1[row][2:]

    parse(belgium_od_table, belgium_od)
    belgium_zones = gpd.read_file(zoning_file)
    #belgium_zones = belgium_zones.astype({'ZONENUMMER': 'int64'})
    belgium_zones = belgium_zones.sort_values(by=['Zonenumber'])
    belgium_zones['centroid_x'] = belgium_zones['geometry'].apply(lambda x: x.centroid.x) # adding additional column?
    belgium_zones['centroid_y'] = belgium_zones['geometry'].apply(lambda x: x.centroid.y)
    return belgium_od, belgium_zones, tot_time_steps


def find_internal_zones_mow(city: str,buffer_D:int, belgium_zones: gpd.geodataframe):
    # queries by name to get internal zone_ids, centroid locations and a geodataframe that just contains the study area
    # TODO: bounding box support could be useful
    R_D = buffer_D
    #R_N = buffer_N
    #SHP = gpd.read_file('Zonering\Zonering_RVM_VLA.SHP')
    shp1 = belgium_zones[belgium_zones['GEMEENTE_L'].isin([city])]  # just to find the location of the city
    X = np.average(shp1.X_LAMB.to_numpy())  # average of all X coordinates in Gemeente of 'place'
    Y = np.average(shp1.Y_LAMB.to_numpy())

    # m1 = 1
    belgium_zones['Dist'] = ((belgium_zones['X_LAMB'] - X) ** 2 + (belgium_zones['Y_LAMB'] - Y) ** 2) ** 0.5
    shp = belgium_zones[belgium_zones.Dist <= R_D * 1000]
    internal_zones = shp
    internal_zone_ids = internal_zones['Zonenumber'] # already an integer
    internal_X = internal_zones['centroid_x'].to_numpy()
    internal_Y = internal_zones['centroid_y'].to_numpy()
    print(f'found {internal_X.size} centroids for {city = }')
    return internal_zone_ids, internal_X, internal_Y, internal_zones

def generate_demand_2d(city: str, buffer_D:int,full_OD_matrix:np.ndarray,belgium_zones_int:gpd.geodataframe,i: np.ndarray, SHP:gpd.geodataframe,C:np.ndarray,buffer_transit:int,buffer_zones:float,ext:int, X1=None, Y1=None, max_study_area_dist=.5):
    """
    generates a local od table from the mow data for a given city
    Parameters
    ----------
    city : str, name of the city
    full_OD_matrix: full OD Matrix already considering the time of day
    X1 : centroid longitude
    Y1 : centroid latitude
    max_study_area_dist : buffer to consider what's inside the study area???

    Returns
    local_od_table, X, Y -------

    """
    #belgium_od_table, belgium_zones, tot_time_steps = load_data_mow()
    belgium_od_table = full_OD_matrix
    #tot_time_steps =

    internal_zone_ids=i
    internal_zones = belgium_zones_int
    X1 = internal_zones['centroid_x'].to_numpy()
    Y1 = internal_zones['centroid_y'].to_numpy()

    X_full_zoning = SHP['centroid_x'].to_numpy()
    Y_full_zoning = SHP['centroid_y'].to_numpy()
    dist, mapping = find_mapping(X1, Y1,X_full_zoning, Y_full_zoning)
    # mapping denotes the ids of the closest internal centroid for each centroid of the zoning file. So size of mapping is 4098
    #is_in_study_area = dist / 1000 < max_study_area_dist # max_study_area_dist in kms, to only find centroids (indexes) which are close enough to our cordon
    is_in_study_area1 = np.isin(SHP['Zonenumber'], internal_zone_ids) # only internal zones (already with a buffer-max study area: not used) are in_study_area
    # if internal zones were provided and are not taken from the data
    # we need to determine which of the zones are in our study a+rea
    # we later discard all traffic for which neither origin nor destination are inside the study area, see fill_od

    SHP['Dist'] = ((SHP['centroid_x'] - C[0]) ** 2 + (SHP['centroid_y'] - C[1]) ** 2) ** 0.5 # Distance of each centroid from centre.
    is_in_transit_buffer =  np.array(SHP.Dist <= buffer_transit * 1000)

    local_od_table = np.zeros(shape=(X1.size, X1.size), dtype=np.float32)

    @njit(parallel=True)
    def fill_od(global_od, local_od, mapping, is_in_study_area): # considers an extra buffer (on top of buffer used for re-saving shapefile) for considering centroids in study area.
       for origin in prange(global_od.shape[0]):
            for destination in range(global_od.shape[0]):
                if is_in_study_area[origin] or is_in_study_area[destination]:
                    local_od[mapping[origin]][mapping[destination]] += global_od[origin][destination]
    def fill_od1(global_od, local_od, mapping, is_in_study_area1): # no extra buffer, only internal zones (already includes a buffer) considered in study area
       for origin in prange(global_od.shape[0]):
            for destination in range(global_od.shape[0]):
                if is_in_study_area1[origin] or is_in_study_area1[destination]:
                    local_od[mapping[origin]][mapping[destination]] += global_od[origin][destination]

    def hull_points(C:np.ndarray,r:float,P:np.ndarray, hb = 70000):
        """
        :param C:
        :param r: buffer of the internal zones/ internal study area.
        :param P:
        :param hb: 70 km - just ensure that it is big enough to contain all transit buffer
        :return:
        """
        from math import sqrt
        PC = P - C
        PCmg = np.linalg.norm(P - C)
        D = sqrt(PCmg ** 2 - r ** 2)  # simple pythagorus th.

        a1 = (r * (PC[0]) + D * PC[1]) / (D ** 2 + r ** 2) # This reduces to -sin(alpha1+alpha2) where alpha 1 is angle of PC with PT11 and alpha 2 is angle of PC with x axis.
        b1 = (r * a1 - PC[0]) / D # This reduces to +cos(alpha 1 + alpha2), alpha1 + alpha 2 becomes the angle of CT11 with y axis.
        A1 = np.array([a1, b1]) # A1 is the unit vector from C to T11. r*A1 will be the vector from C to T11
        T11 = C + r * A1
        b2 = (D * (PC[0]) + r * PC[1]) / (D ** 2 + r ** 2)
        a2 = (PC[0] - b2 * D) / r
        A2 = np.array([a2, b2])
        T12 = C + r * A2

        # Next 2 points
        #hb =   # just the distance along the tangent from the tangent point, it doesn't matter if it goes beyond transit_buffer
        # as irrespective of how much it goes outwards i.e. irrespective of how big the hull is we will only work with centroids
        # that are in the hull and within the buffer_transit and not in the internal zones (buffer_zones)
        pcap1 = (T11 - P) / np.linalg.norm(T11 - P)  # unit vector in the direction of new point along the tangent
        pcap2 = (T12 - P) / np.linalg.norm(T12 - P)

        T13 = T11 + hb * pcap1
        T14 = T12 + hb * pcap2

        T = np.concatenate(( [T11], [T12], [T13], [T14]), axis=0)
        return T

    def in_hull(points:np.ndarray, hull_points:np.ndarray):
        """
        :param points: POINTS TO BE TESTED
        :param hull_points: points defining the hull
        :return: True/false list of points
        """
        import numpy as np
        from scipy.spatial import ConvexHull

        hull = ConvexHull(hull_points)
        tolerance = 1e-12
        # get array of boolean values indicating in hull if True
        inside_hull = np.all(np.add(np.dot(points, hull.equations[:, :-1].T), hull.equations[:, -1]) <= tolerance, axis=1)
        return inside_hull


    def fill_od_tr(global_od, local_od, mapping, is_in_study_area1, X_full_zoning,Y_full_zoning,dist,C, buffer_transit, buffer_zones): # As a topup on fill_od1
       Destination_points = np.array([X_full_zoning, Y_full_zoning])
       Destination_points = np.transpose(Destination_points) # all centroid locations to check whether a particular D lies within the hull
       for origin in prange(global_od.shape[0]):
           if ~is_in_study_area1[origin] and is_in_transit_buffer[origin]:
               P = np.array([X_full_zoning[origin], Y_full_zoning[origin]]) #origin coordinates
               points_for_hull = hull_points(C,buffer_zones*1000,P)
               is_in_hull = in_hull(Destination_points,points_for_hull)# a convex hull for each origin
               for destination in range(global_od.shape[0]): # all centroids
                   if all([~is_in_study_area1[destination], is_in_hull[destination], is_in_transit_buffer[destination]]): # only possible transit destinations
                       if global_od[origin][destination]>1e-3: # so no calculation if demand is anyway = 0
                           Omap_dist = dist[origin] / 1000  # converted to kms for next step
                           Dmap_dist = dist[destination] / 1000
                           OD_dist = (((X_full_zoning[origin] - X_full_zoning[destination]) ** 2 + (
                                   Y_full_zoning[origin] - Y_full_zoning[destination]) ** 2) ** 0.5) / 1000
                           F = 1 - ((1 - 0.1) / (buffer_transit - buffer_zones)) * ((Omap_dist + Dmap_dist) / 2)  # 0.1 at x = buffer_transit-buffer_zones (transit boundary), 1 at x = 0 (internal zones boundary)
                           local_od[mapping[origin]][mapping[destination]] += F * global_od[origin][destination]





    def fill_od2(global_od, local_od, mapping, X_full_zoning,Y_full_zoning,dist): # considers all centroids, doesn't throw away anything but scales all external demand with f
        for origin in prange(global_od.shape[0]):
            for destination in range(global_od.shape[0]):
                Omap_dist = dist[origin]
                Dmap_dist = dist[destination]
                OD_dist =((X_full_zoning[origin]-X_full_zoning[destination])**2 + (Y_full_zoning[origin]-Y_full_zoning[destination])**2)**0.5
                f = max(0, 1 - 0.3*(min(Omap_dist, Dmap_dist)/OD_dist)) # really bad f, doesn't make sense
                local_od[mapping[origin]][mapping[destination]] += f*global_od[origin][destination]

    # the above implementation has the following shortcomings:
    # 1. Trips going from inside the study area to the outside are just mapped to a single
    # location on the boundary based on straight line distance. It's clear one ideally would
    # take into account available routes in this mapping.
    # 2. Similar to one, traffic going from the outside of the study area to the inside should be
    # dispersed among the available routes and not mapped to a single point.
    # 3. Traffic that we discard because neither origin nor destination are inside the study area may actually
    # have an impact because a viable route crosses our study area.
    #fill_od(belgium_od_table, local_od_table, mapping, is_in_study_area1)

    fill_od1(belgium_od_table, local_od_table, mapping, is_in_study_area1) # FOR EXTERNAL W/O TRANSIT
    if ext==2: # top up to add transit on local od table w/o transit
        fill_od_tr(belgium_od_table, local_od_table, mapping, is_in_study_area1, X_full_zoning,Y_full_zoning,dist,C,buffer_transit,buffer_zones)

    #fill_od2(belgium_od_table, local_od_table, mapping, X_full_zoning,Y_full_zoning,dist)
    #local_od_table_static = local_od_table[:, :, time_of_day]
    print(f'Demand successfully extracted for {city=}')

    out_proj, in_proj = 'epsg:4326', 'epsg:31370'
    transformer = Transformer.from_crs(in_proj, out_proj)
    Y, X = transformer.transform(X1,Y1)
    return local_od_table, X, Y


def generate_demand(city: str, buffer_D:int,time_of_day:int, X1=None, Y1=None, max_study_area_dist=.5):
    """
    generates a local od table from the mow data for a given city
    Parameters
    ----------
    city : str, name of the city
    X1 : centroid longitude
    Y1 : centroid latitude
    max_study_area_dist : buffer to consider what's inside the study area???

    Returns
    local_od_table, X, Y -------

    """
    belgium_od_table, belgium_zones, tot_time_steps = load_data_mow()
    if X1 is None:
        assert Y1 is None
        zoning_provided = False
        internal_zone_ids, X1, Y1, internal_zones = find_internal_zones_mow(city,buffer_D, belgium_zones)
    else: # this is not executed I think as X1 and Y1 are not provided explicitly by the user
        zoning_provided = True
        warn('not tested for validity ..., see comments below before proceeding')
        in_proj, out_proj = 'epsg:4326', 'epsg:31370'  # from lat lon to belge_lambert
        transformer = Transformer.from_crs(in_proj, out_proj)
        X1, Y1 = transformer.transform(Y1, X1)  # pyproj returns (lon,lat), don't quite get why.
        # it would run to pass on your own centroids ..but I reckon we need some dispersion
        # for traffic that has as its destination or origin one of the original centroids that are in the study area
        # e.g. the mapping function below needs to be extended, may disperse based
        # on the distance distribution of the k nearest
        # centroids(?) see query method in find_mapping.
    X_full_zoning = belgium_zones['centroid_x'].to_numpy()
    Y_full_zoning = belgium_zones['centroid_y'].to_numpy()
    dist, mapping = find_mapping(X1, Y1,X_full_zoning, Y_full_zoning)
    # mapping denotes the ids of the closest internal centroid for each centroid of the zoning file. So size of mapping is 4098
    #is_in_study_area = dist / 1000 < max_study_area_dist # max_study_area_dist in kms, to only find centroids (indexes) which are close enough to our cordon
    is_in_study_area1 = np.isin(belgium_zones['Zonenumber'], internal_zone_ids) # only internal zones (already with a buffer) are in_study_area
    # if internal zones were provided and are not taken from the data
    # we need to determine which of the zones are in our study a+rea
    # we later discard all traffic for which neither origin nor destination are inside the study area, see fill_od
    local_od_table = np.zeros(shape=(X1.size, X1.size, tot_time_steps), dtype=np.float32)

    @njit(parallel=True)
    def fill_od(global_od, local_od, mapping, is_in_study_area): # considers an extra buffer (on top of buffer used for re-saving shapefile) for considering centroids in study area.
       for origin in prange(global_od.shape[0]):
            for destination in range(global_od.shape[0]):
                if is_in_study_area[origin] or is_in_study_area[destination]:
                    for t in range(tot_time_steps):
                        local_od[mapping[origin]][mapping[destination]][t] += global_od[origin][destination][t]
    def fill_od1(global_od, local_od, mapping, is_in_study_area1): # no extra buffer, only internal zones (already includes a buffer) considered in study area
       for origin in prange(global_od.shape[0]):
            for destination in range(global_od.shape[0]):
                if is_in_study_area1[origin] or is_in_study_area1[destination]:
                    for t in range(tot_time_steps):
                        local_od[mapping[origin]][mapping[destination]][t] += global_od[origin][destination][t]

    def fill_od2(global_od, local_od, mapping, X_full_zoning,Y_full_zoning,dist): # considers all centroids, doesn't throw away anything but scales all external demand with f
        for origin in prange(global_od.shape[0]):
            for destination in range(global_od.shape[0]):
                for t in range(tot_time_steps):
                    Omap_dist = dist[origin]
                    Dmap_dist = dist[destination]
                    OD_dist =((X_full_zoning[origin]-X_full_zoning[destination])**2 + (Y_full_zoning[origin]-Y_full_zoning[destination])**2)**0.5
                    f = max(0, 1 - 0.3*(min(Omap_dist, Dmap_dist)/OD_dist))
                    local_od[mapping[origin]][mapping[destination]][t] += f*global_od[origin][destination][t]

    # the above implementation has the following shortcomings:
    # 1. Trips going from inside the study area to the outside are just mapped to a single
    # location on the boundary based on straight line distance. It's clear one ideally would
    # take into account available routes in this mapping.
    # 2. Similar to one, traffic going from the outside of the study area to the inside should be
    # dispersed among the available routes and not mapped to a single point.
    # 3. Traffic that we discard because neither origin nor destination are inside the study area may actually
    # have an impact because a viable route crosses our study area.
    #fill_od(belgium_od_table, local_od_table, mapping, is_in_study_area1)
    fill_od1(belgium_od_table, local_od_table, mapping, is_in_study_area1)
    #fill_od2(belgium_od_table, local_od_table, mapping, X_full_zoning,Y_full_zoning,dist)
    local_od_table_static = local_od_table[:, :, time_of_day]
    print(f'Demand successfully extracted for {city=}')
    if zoning_provided:
        return local_od_table_static
    else:
        out_proj, in_proj = 'epsg:4326', 'epsg:31370'
        transformer = Transformer.from_crs(in_proj, out_proj)
        Y, X = transformer.transform(internal_zones['centroid_x'].to_numpy(),internal_zones['centroid_y'].to_numpy())
        return local_od_table_static, X, Y


if __name__ == '__main__':
    generate_demand('Gent')
