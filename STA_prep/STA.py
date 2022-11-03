


import numpy as np
import pandas as pd
import re
from numba import njit, prange
import geopandas as gp
import os
import osmnx as ox
from extract_demand import generate_demand
from extract_demand import generate_demand_2d



def STA_initial_setup(place: str, buffer_D:int, buffer_N:int,buffer_transit:int, D_sup:float, time_of_day:int,ext: int):
    '''
    :param place: string for municipality IN CAPITALS
    :param buffer: buffer (non-negative integer) in kms from the centroid of entire municipality
    :param time_of_day: integer from 1 to 25
    :param ext: int 0 for only internal OD matrix, ~=0 for external demand projected to boundary (only if either O or D is internal)
    :return: osm file for full road network in the buffer, shape file for zones in the buffer, od matrix for zones in the buffer
    **BE CAUTIOUS ABOUT SOME PARTIALLY OF ILL LOADED FILES FROM FAILED RUNS OF PREVIOUS TIME MAY LIE IN THE FILE PATHS AND
    PREVENT THE FUNCTION FROM SAVING THE CORRECT FILES**
    '''
    R_D = buffer_D
    R_N = buffer_N
    SHP = gp.read_file('shapefile_data/Full projected centroids/Full projected centroids.shp')
    #SHP = gp.read_file('Zonering\Zonering_RVM_VLA.SHP') # full zoning file
    shp1 = SHP[SHP['GEMEENTE_L'].isin([place])] # just to find the location of the city
    X = np.average(shp1.centroid_x.to_numpy())  # average of all X coordinates in Gemeente of 'place'
    Y = np.average(shp1.centroid_y.to_numpy())
    C = np.array([X,Y])

    # READING THE OVERALL SHAPEFILE and SAVING THE SHAPEFILE ONLY WITH ZONES WITHIN THE BUFFER

    shp_name = place + '_' + str(R_D)+ '_' + str(int(D_sup*10)) # D_sup*10 is only to avoid the decimal point in file name. NO OTHER SIGNIFICANCE
    filepath_shp = os.path.join('shapefile_data',
                                str(shp_name))  # this is the directory where the shape file and other accompanying files will be saved
    zoning_file = os.path.join(filepath_shp, shp_name + '.shp')  # the particular .shp file
    m1=0

    if not os.path.isfile(zoning_file):  # only if such a file doesn't exist already
        # SHP = gp.read_file('Zonering\Zonering_RVM_VLA.SHP')
        m1=1
        SHP['Dist'] = ((SHP['centroid_x'] - X) ** 2 + (SHP['centroid_y'] - Y) ** 2) ** 0.5
        shp = SHP[SHP.Dist <= (R_D+D_sup) * 1000] # internal zones
        shp.to_file(filepath_shp, encoding='UTF-8')

    if m1 ==0:
        print('shapefile has been found as ' + filepath_shp)
    else:
        print('shapefile has been saved as ' + filepath_shp)



    # EXTRACTING THE NETWORK WITH THE BUFFER

    network_name = place + '_' + str(R_N) + '.osm'
    filepath_osm = os.path.join('network_data', str(network_name))
    m2=0

    if not os.path.isfile(filepath_osm):
        m2=1
        full_g = ox.graph_from_place(place, network_type='drive', buffer_dist=R_N * 1000)
        ox.save_graph_xml(full_g, filepath=filepath_osm)

    if m2 ==0:
        print('osmfile has been found as ' + filepath_osm)
    else:
        print('osmfile has been saved as ' + filepath_osm)

    # FULL OD MATRIX
    # check if the full matrix exists for the time of the day
    od_matrix_name = 'full_belgium_od_matrix_' + str(time_of_day) + '.txt'
    filepath_full_od_matrix = os.path.join('od_matrix_data', 'full_matrices', str(od_matrix_name))
    m4 = 0

    if not os.path.isfile(filepath_full_od_matrix):  # if file not already present- for the time of day
        m4 = 1
        dynamic_od_table_csv = ("Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.CSV")
        belgium_od_table = pd.read_csv(dynamic_od_table_csv, header=2)
        belgium_od_table.columns = belgium_od_table.columns.str.replace('\'', '')
        tot_time_steps = [int(e) for e in re.split("[^0-9]", belgium_od_table.columns[-1]) if e != ''][0]
        tot_destinations = belgium_od_table['B'].max()
        tot_origins = belgium_od_table['H'].max()
        belgium_od_matrix = np.zeros((tot_origins, tot_destinations, tot_time_steps), dtype=np.float32)
        belgium_od_table = belgium_od_table.to_numpy()

        # make copying fast with numba...
        @njit(parallel=True)
        def parse(matrix1, matrix2):
            for row in prange(matrix1.shape[0]):
                i = np.uint32(matrix1[row][0] - 1)
                j = np.uint32(matrix1[row][1] - 1)
                matrix2[i][j] = matrix1[row][2:]

        parse(belgium_od_table, belgium_od_matrix)  # this gives us full OD matrix for 24 hours

        belgium_od_matrix = belgium_od_matrix[:, :, time_of_day - 1]
        np.savetxt(filepath_full_od_matrix, belgium_od_matrix, fmt='%0.4f')
    else:  # if full matrix present just read it
        belgium_od_matrix = np.genfromtxt(filepath_full_od_matrix)
    if m4 == 0:
        print('full_od_matrix has been found as ' + filepath_full_od_matrix)
    else:
        print('full_od_matrix has been saved as ' + filepath_full_od_matrix)

    # REDUCED OD MATRIX

    if ext ==0:
        # 1st way only internal demand
        # Check if the reduced matrix exists
        reduced_od_matrix_name = place + '_' + str(R_D) + '_' + str(time_of_day) + '_' + ".xlsx"
        filepath_od_matrix = os.path.join('od_matrix_data', str(reduced_od_matrix_name))
        m3 = 0
        if not os.path.isfile(filepath_od_matrix):  # if reduced matrix not already present, then we go further
            m3 = 1
        # Now we go towards reduced matrix but remember this is only if reduced matrix didn't exist initially
            belgium_zones = gp.read_file(zoning_file)  # we read the shp file saved in previous steps
            belgium_zones = belgium_zones.sort_values(by=['Zonenumber'])  # No conversion needed as Zonenumber is integer
            # in the original file
            # This is vital in order to arrange the randomly numbered remaining zones in an increasing order. Why? Because when we
            # will load the shape file in Visum, the zonenumber field will become the 'number' [REMEMBER TO DO SO WHILE IMPORTING IN VISUM]
            # and the OD matrix must correspond to the same order.

            i = belgium_zones['Zonenumber']
            i = i.to_numpy()
            belgium_od_matrix_trimmed = belgium_od_matrix[i - 1, :]
            belgium_od_matrix_trimmed = belgium_od_matrix_trimmed[:, i - 1]

            # belgium_od_matrix_trimmed_static = belgium_od_matrix_trimmed[:, :, time_of_day]

            # np.savetxt(filepath_od_matrix, belgium_od_matrix_trimmed_static, fmt='%0.4f')
            ## convert your array into a dataframe
            df = pd.DataFrame(belgium_od_matrix_trimmed)
            df.to_excel(filepath_od_matrix,
                        index=False)  # VERY SLOW PROCESS! MANUAL IS SLOW AS WELL FOR 2720 by 2720 matrix, may take around 20 minutes

        if m3 == 0:
            print('reduced_od_matrix has been found as ' + filepath_od_matrix)
        else:
            print('reduced_od_matrix has been saved as ' + filepath_od_matrix)

    elif ext ==1:
        # 2nd way with external demand projected to boundaries. Still no transit demand
        od_matrix_ext_name = place + '_ext' + '_' + str(R_D) + '_' + str(time_of_day) +  '_' + str(int(D_sup*10))+ ".xlsx"
        filepath_od_matrix_ext = os.path.join('od_matrix_data_ext', str(od_matrix_ext_name))
        m5 = 0
        if not os.path.isfile(filepath_od_matrix_ext):  # if file not already present- for the time of day
            m5 = 1
            belgium_zones_int = gp.read_file(zoning_file)  # we read the shp file saved in previous steps, the reason we don't use shp directly is because shp is not always executed
            belgium_zones_int = belgium_zones_int.sort_values(by=['Zonenumber'])
            i = belgium_zones_int['Zonenumber']
            i = i.to_numpy()
            SHP = SHP.sort_values(by=['Zonenumber'])
            od_table_ext, X, Y = generate_demand_2d(place, buffer_D,belgium_od_matrix,belgium_zones_int,i,SHP,C,buffer_transit, R_D+D_sup,ext) #2d function when time of day is already accounted for and no need to find internal zones- for fast execution
            df = pd.DataFrame(od_table_ext)
            df.to_excel(filepath_od_matrix_ext, index=False)
        if m5 == 0:
            print('reduced_od_matrix_ext has been found as ' + filepath_od_matrix_ext)
        else:
            print('reduced_od_matrix_ext has been saved as ' + filepath_od_matrix_ext)

    elif ext==2:
        # 3rd way with external demand projected to boundaries. Transit demand included w/o elasticity
        od_matrix_ext_tr_name = place + '_ext_tr' + '_' + str(R_D) + '_' + str(time_of_day) + '_' + str(int(D_sup*10))+ ".xlsx"
        filepath_od_matrix_ext_tr = os.path.join('od_matrix_data_ext_tr', str(od_matrix_ext_tr_name))
        m6 = 0
        if not os.path.isfile(filepath_od_matrix_ext_tr):  # if file not already present- for the time of day
            m6 = 1
            belgium_zones_int = gp.read_file(
                zoning_file)  # we read the shp file saved in previous steps, the reason we don't use shp directly is because shp is not always executed
            belgium_zones_int = belgium_zones_int.sort_values(by=['Zonenumber'])
            i = belgium_zones_int['Zonenumber']
            i = i.to_numpy()
            SHP = SHP.sort_values(by=['Zonenumber'])
            od_table_ext_tr, X, Y = generate_demand_2d(place, buffer_D, belgium_od_matrix, belgium_zones_int, i,
                                                    SHP,C,buffer_transit, R_D+D_sup,ext)  # 2d function when time of day is already accounted for and no need to find internal zones- for fast execution
            df = pd.DataFrame(od_table_ext_tr)
            df.to_excel(filepath_od_matrix_ext_tr, index=False)
        if m6 == 0:
            print('reduced_od_matrix_ext_tr has been found as ' + filepath_od_matrix_ext_tr)
        else:
            print('reduced_od_matrix_ext_tr has been saved as ' + filepath_od_matrix_ext_tr)

    print(
        'All three/four files have been saved (if not already present): osm file for network, reduced shapefile, reduced OD Matrix in excel')


















