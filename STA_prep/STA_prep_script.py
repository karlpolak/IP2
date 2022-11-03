import STA
place = "BRUSSEL" # CAPS
buffer_N = 3
buffer_D = buffer_N
buffer_transit = 45
D_sup=1 # kms, use in the range 0-1 (res:0.1) if you want to include some boundary zones
time_of_day = 9
ext = 2 # 0 for only internal, 1 for taking external demand w/o transit, 2 for external with transit
STA.STA_initial_setup(place, buffer_D,buffer_N,buffer_transit, D_sup, time_of_day, ext)
