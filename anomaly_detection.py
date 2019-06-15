# ******************************************************************************
# anomaly_detection.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/12/19   Paudel     Initial version,
# ******************************************************************************

# anomaly detection based on Isolation Forest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc
from sklearn import preprocessing
from sklearn import metrics

#anomaly detection based on Robust Random Cut Forest
import rrcf

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np


class AnomalyDetection:

    def __init__(self):
        # print("\n\n------ Start Anomaly Detection ---- ")
        pass

    def read_csv_file(self, csv_file):
        tcp = pd.DataFrame(index=[], columns=[])
        tcp1 = pd.read_csv(csv_file)
        tcp = tcp.append(tcp1, ignore_index=True)
        tcp = tcp.iloc[:, [1, 2, 3, 4]]
        tcp.columns = ['source', 'destination', 'anomaly', 'hours_past']
        return tcp

    def read_sketch(self, file_name):
        sketches = []
        with open(file_name) as infile:
            lines = infile.readlines()
            for line in lines:
                sketch = []
                for n in line.strip('\n').split(','):
                    sketch.append(n)
                sketches.append(sketch)
        return np.array(sketches).astype(np.float64)

    def get_top_k_anomalies(self, graphs, k):
        top_k_index = []
        for index, row in graphs.nlargest(k, 'anomaly').iterrows():
            top_k_index.append(index)
        return top_k_index

    def get_top_k_performance(self, top_k, true_anomalies, predicted_anomalies, algo):
        if algo == 'iso':
            print("\n\n--- Performance on (K = ", len(top_k) , " ) using Isolation Forest")
        elif algo == 'rrcf':
            print("\n\n--- Performance on (K = ", len(top_k), " ) using Robust Random Cut Forest")

        k_true = []
        k_predicted = []

        for index in top_k:
            #print(index)
            k_true.append(true_anomalies[index])
            k_predicted.append(predicted_anomalies[index])

        #print("Top K: ", top_k)
        #print("True : ", k_true)
        #print("Predicted : ", k_predicted)

        target_names = ['Normal', 'Anomaly']
        print(metrics.classification_report(k_true, k_predicted, target_names=target_names))

    def robust_random_cut(self, sketch_vector):
        # Set tree parameters
        # Specify sample parameters

        forest = []
        num_trees = 50
        tree_size = 256
        n = len(sketch_vector)
        sample_size_range = (n // tree_size, tree_size)
        while len(forest) < num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n, size=sample_size_range,
                                    replace=False)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(sketch_vector[ix], index_labels=ix)
                     for ix in ixs]
            forest.extend(trees)

        # Compute average CoDisp
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index

        predicted = avg_codisp > avg_codisp.quantile(0.70)
        # print("Predicted: ", predicted)
        return predicted


    def isolation_forest(self, sketch_vector):
        vectors = preprocessing.scale(sketch_vector)
        clf = IsolationForest(n_estimators=100, max_samples=100, contamination=0.9, behaviour="new")
        clf.fit(vectors)
        predicted = clf.score_samples(vectors)
        predicted = predicted > np.quantile(predicted,0.70)
        return predicted


    def anomaly_detection(self, sketch_vector, args):
        '''
        :param sketch_vector:
        :param args:
        :return:
        '''
        if args.anom_algo == 'iso':
            predicted = self.isolation_forest(sketch_vector=sketch_vector)
        elif args.anom_algo == 'rrcf':
            predicted = self.robust_random_cut(sketch_vector)

        # print("Predicted: ", predicted)
        p_anom = []
        predicted_anomalies = []
        index = 0
        for i in predicted:
            predicted_anomalies.append(i*1)
            if i == 1:
                p_anom.append(index)
            index += 1


        # print("Anomaly: ", tcp.loc[tcp['anomaly']!=0])
        # groupby hour_past, timestamps which contain more than 1000 anomalous communication are anomalous timestamps

        tcp = self.read_csv_file(args.csv_file)
        truth = tcp.groupby('hours_past').sum()

        truth = truth.reset_index()
        top_k = self.get_top_k_anomalies(truth, 300)
        top_k.sort()

        true_anomalies = ((truth.anomaly.values > 50) * 1)

        self.get_top_k_performance(top_k, true_anomalies, predicted_anomalies, args.anom_algo)

        #true_anomaly = truth[truth['anomaly']>100]
        '''
        t_anom = []
        index = 0
        for i in truth
            if i == 1:
                t_anom.append(index)
            index += 1

        print("Predicted Anomaly: ", p_anom)
        print("True Anomaly: ", t_anom)
        # print("Truth:  \n\n", len(truth), truth)
        target_names = ['Normal', 'Anomaly']
        print(metrics.classification_report(truth, predicted, target_names=target_names))
        precision, recall, thresholds = precision_recall_curve(truth, predicted)
        print('Area Under Curve:', auc(recall, precision))

        # matplotlib inline
        # plt.plot(recall, precision, marker='.')
        # #plt.plot(detected, marker='.')
        # plt.show()'''
