# ******************************************************************************
# preprocessing.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/13/19   Paudel     Initial version,
# ******************************************************************************

import graph_utils
import os, glob
import pandas as pd

class Preprocess():

    def __init__(self):
        pass

    def preprocess_gfiles(self, data_folder):
        files = glob.glob(data_folder)
        graphs = {}
        id = 1
        for file in files:
            graphs[id] = graph_utils.read_graph(file)
            id += 1
        return graphs


    def preprocess_gexf(self, data_folder):
        graphs = {}
        for root, dirs, files in os.walk(data_folder):
            index = 1
            for name in files:
                graphs[index] = graph_utils.get_gex_graph(os.path.join(root, name))
                index += 1

        return graphs

    def preprocess_single_gfile(self, file_name):
        return graph_utils.read_send_gfiles(file_name)

    def parse_tcp_dump(self, data_folder, csv_file):
        # import tcpdump files
        tcp = pd.DataFrame(index=[], columns=[])
        files = glob.glob(data_folder)

        for file in files:
            print("File: ", file)
            tcp1 = pd.read_csv(file, delim_whitespace=True, header=None)
            tcp = tcp.append(tcp1, ignore_index=True)

        tcp = tcp.iloc[:, [1, 2, 7, 8, 9]]
        tcp.columns = ['date', 'time', 'source', 'destination', 'anomaly']
        # print (tcp)
        # exclude collapsing data
        tcp = tcp[tcp.date != '07/32/1998']

        # acquire datetime information
        tcp['date_time'] = tcp['date'] + '/' + tcp['time']
        tcp = tcp.drop('date', axis=1)
        tcp = tcp.drop('time', axis=1)
        tcp['date_time'] = pd.to_datetime(tcp['date_time'], format='%m/%d/%Y/%H:%M:%S')

        # calculate how many hours passed since the initial time
        initial_time = tcp['date_time'].min()
        tcp['date_time'] = tcp['date_time'] - initial_time
        tcp['hours_past'] = tcp['date_time'].dt.days * 24 + tcp['date_time'].dt.seconds // 3600
        tcp = tcp.drop('date_time', axis=1)
        tcp = tcp.sort_values('hours_past')
        graphs = tcp.loc[:, ['source', 'destination', 'hours_past']]
        graphs = graphs.values
        tcp.to_csv(csv_file)
        return graphs, tcp