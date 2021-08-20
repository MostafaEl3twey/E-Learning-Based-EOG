import csv
from csv import writer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter, lfilter
from itertools import zip_longest

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerows(list_of_elem)
        write_obj.close()
def totalcount(name,counter_UP,counter_Down,counter_center,counter_blink):
    listofUP = ['Total UP', counter_UP]
    list1 = [None]
    list = [list1, listofUP]
    upCount = zip_longest(*list, fillvalue='')
    append_list_as_row(name+'/Up.csv', upCount)

    listofDown = ['Total Down', counter_Down]
    list = [list1, listofDown]
    DownCount = zip_longest(*list, fillvalue='')
    append_list_as_row(name+'/Down.csv', DownCount)

    listofcenter = ['Total Center', counter_center]
    list = [list1, listofcenter]
    centerCount = zip_longest(*list, fillvalue='')
    append_list_as_row(name+'/Center.csv', centerCount)

    listofblink = ['Total Blink', counter_blink]
    list = [list1, listofblink]
    blinkCount = zip_longest(*list, fillvalue='')
    append_list_as_row(name+'/Blink.csv', blinkCount)
def truncate(name):
    fileVariable = open(name + '/' + name + '_Final.csv', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
def Main(name):
    truncate(name)

    'Reading'
    path_V= name+'/V_'+name+'.csv'
    path_H = name + '/H_' + name + '.csv'
    V = (pd.read_csv(path_V, header=None))
    H = (pd.read_csv(path_H, header=None))
    V_Values = V[0][:].to_numpy()
    H_Values = H[0][:].to_numpy()
    ############################################################

    Counter = 0
    list_of_time = []
    value_of_V = []
    value_of_H = []
    ############################################################

    for a, x in enumerate(V_Values):
        if (x > 1.8):
            value_of_V.append(x)
            list_of_time.append(a)
            if (V_Values[a + 1] < 1.8):
                for h in list_of_time:
                    value_of_H.append(H_Values[h])
                time_diffrence = list_of_time[-1] - list_of_time[0]
                if (time_diffrence < 500):
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Blink', time_diffrence, '***************']
                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]
                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)
                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
                else:
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter,'UP', time_diffrence, '***************']

                    d = [line0,line1,line0,list_of_time, value_of_V,value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/'+name+'_Final.csv', export_data)

                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
        elif (x < 1.55):
            value_of_V.append(x)
            list_of_time.append(a)
            if (V_Values[a + 1] > 1.55):
                for h in list_of_time:
                    value_of_H.append(H_Values[h])
                time_diffrence = list_of_time[-1] - list_of_time[0]
                Counter += 1
                line0 = [None]
                line1 = [Counter, 'Down', time_diffrence, '***************']
                d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                export_data = zip_longest(*d, fillvalue='')
                append_list_as_row(name + '/' + name + '_Final.csv', export_data)

                list_of_time = []
                value_of_V = []
                value_of_H = []
        else:
            value_of_V.append(x)
            list_of_time.append(a)
            if (x == V_Values[-1]):
                for h in list_of_time:
                    value_of_H.append(H_Values[h])
                time_diffrence = list_of_time[-1] - list_of_time[0]
                if(value_of_H[0]>2):
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Right', time_diffrence, '***************']

                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)


                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
                elif(value_of_H[0]<1):
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Left', time_diffrence, '***************']

                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)



                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
                else:
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Center', time_diffrence, '***************']

                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)



                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
            elif (V_Values[a + 1] < 1.55 or V_Values[a + 1] > 1.8):
                for h in list_of_time:
                    value_of_H.append(H_Values[h])
                time_diffrence = list_of_time[-1] - list_of_time[0]
                if(value_of_H[0]>2):
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Right', time_diffrence, '***************']

                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)


                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
                elif(value_of_H[0]<1):
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Left', time_diffrence, '***************']

                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)



                    list_of_time = []
                    value_of_V = []
                    value_of_H = []
                else:
                    Counter += 1
                    line0 = [None]
                    line1 = [Counter, 'Center', time_diffrence, '***************']

                    d = [line0, line1, line0, list_of_time, value_of_V, value_of_H]

                    export_data = zip_longest(*d, fillvalue='')
                    append_list_as_row(name + '/' + name + '_Final.csv', export_data)


                    list_of_time = []
                    value_of_V = []
                    value_of_H = []

Main('Ehab')

