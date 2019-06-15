# ******************************************************************************
# main.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/5/19   Paudel     Initial version,
# ******************************************************************************

import argparse
from shingle_sketch import ShingleSketch
from anomaly_detection import AnomalyDetection
from spotlight import SpotLight
from preprocessing import Preprocess
import graph_utils
import glob
import numpy as np

def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser("GraphStreamAnomaly")

    args.add_argument("-f","--gexffile", default = "data/test/",
                      help="Path to directory containing gexf files")

    args.add_argument('-g','--graph_folder', default='data/tcpgraph/*',
                      help='graph folder that contains all files (for Shingle)')

    args.add_argument('-c', "--tcp_folder", default="data/tcp/*",
                      help="Path to csv data file ")

    args.add_argument('-t',"--csv_file", default="data/tcp_6week.csv",
                      help="Timestamp for each graph")

    args.add_argument('-v', "--sketch_vector", default="sh4_vec50.txt",
                      help="Timestamp for each graph")

    args.add_argument('-a', "--anom_algo", default="rrcf",
                      help="Algorithm for Anomaly Detection [iso, rrcf]")

    args.add_argument('-s', "--sketch_size", default=50, type=int,
                      help="Sketch Vector Size")

    args.add_argument('-w', "--win_size", default=50, type=int,
                      help="Sliding Window Size")

    args.add_argument('--N', default=10, type=int, help='N time edge count is the length of random walk ')

    args.add_argument('--k_shingle', default=4, type=int, help='Lenght of a shinle')

    return args.parse_args()

def run_spotlight(args, graphs):
    '''
    :param args:
    1. args: list of argument
    :return: None:
    '''
    # sketching graph using spotlight algorithm.
    SL = SpotLight(graphs)
    skvector = SL.sketch(50, 0.2, 0.2)
    np.savetxt(args.sketch_vector, skvector, delimiter=',')

def run_shingle(args, graphs, is_gexf):
    sk = ShingleSketch()
    sk.shingle_sketch(graphs, args, is_gexf)


def main(args):
    pp = Preprocess()

    # ---- Run Spotlight ----
    # graphs = pp.parse_tcp_dump(args.tcp_folder, args.csv_file)

    # tcp = ad.read_csv_file(args.csv_file)
    # graphs = tcp.iloc[:, [0, 1, 3]]
    # graphs.columns = ['source', 'destination',  'hours_past']
    #
    # run_spotlight(args, np.array(graphs))

    # # ---- Run Shingle Sketch -----
    # graph_utils.create_graphs(args.csv_file, args.graph_folder)
    is_gexf = False
    graphs = pp.preprocess_gfiles(args.graph_folder)

    # #--- For Muta or Chemical Data ----
    # graphs = pp.preprocess_gexf(args.gexffile)
    # is_gexf = True
    #
    # #---For DOS Attack Data ---
    # graphs = pp.preprocess_single_gfile("data/dos.g")

    run_shingle(args, graphs, is_gexf)


    ad = AnomalyDetection()
    skvector = ad.read_sketch(args.sketch_vector)
    print(skvector.shape)
    ad.anomaly_detection(skvector, args)


if __name__=="__main__":
    args = parse_args()
    main(args)