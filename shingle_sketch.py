# ******************************************************************************
# shingle_sketch.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 5/15/19   Paudel     Initial version,
# ******************************************************************************
import gensim.models.doc2vec as doc
import random, operator
import graph_utils
from matplotlib import pyplot
import numpy as np
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.algorithms.traversal.breadth_first_search import bfs_tree, bfs_edges


class ShingleSketch():
    win_shingles = {}
    win_sketch = []

    def __init__(self):
        pass

    def arr2str(self, arr):
        result = ""
        for i in arr:
            result += " " + str(i)
        return result

    def get_win_total(self):
        '''
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # NAME: getTotalCount
        #
        # INPUTS: ()
        #
        # RETURN:
        #
        # PURPOSE: Get the total count of shingle in a window
        #
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        :return:
        '''
        total = 0
        for s in self.win_shingles:
            for arr in self.win_shingles[s]:
                total += arr[0]

        return total

    def calculate_similarity(self, shingles):
        match_count = 0
        win_count = self.get_win_total()
        # print("\n\nWindow Total: ", win_count)
        s_count = sum(shingles[i] for i in shingles)
        # print("Shingle Count: ", s_count)

        for s1 in shingles.keys():
            if s1 in self.win_shingles.keys():
                w_count = sum(g_count[0] for g_count in self.win_shingles[s1])
                #print("Match: ", s1, shingles[s1], win_count)
                if shingles[s1] < w_count:
                    match_count += shingles[s1]
                else:
                    match_count += w_count
        # print("Match Count: ", match_count, win_count+s_count-match_count)
        # jaccard = (match_count/(win_count+s_count-match_count))
        # print("Jaccard: ", jaccard)
        return (match_count/(win_count+s_count-match_count))

    def random_walk(self, G, walk_len):
        walkList = []
        node_ids = []
        for n in G.nodes(data=True):
            node_ids.append(n)
        #print(node_ids)
        try:
            curr_node = random.choice(node_ids)
            curr_node = curr_node[0]
            while (len(walkList) < walk_len):
                walkList.append(G.node[curr_node]['label'])
                curr_node = random.choice(list(G.neighbors(curr_node)))
        except:
            pass
        return walkList

    def generate_shingles(self, walk_path, walk_len, k_shingle):
        shingles = {}
        i = 0
        while (i < len(walk_path)-k_shingle):
            shingle = walk_path[i]
            for j in range(1, k_shingle):
                shingle = shingle + '-' + walk_path[i+j]
            if shingle not in shingles:
                shingles[shingle] = 1
            else:
                freq = shingles[shingle]
                shingles[shingle] = freq + 1
            i += 1
        return shingles

    def update_chunk(self, s, graph_count, param_w):
        '''
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # NAME: update_shingles_frequency()
        #
        # INPUTS: (s) Shingle List with count of instances for current window
        #
        # RETURN: ()
        #
        # PURPOSE: Maintain the list of shingle and their frequency in the window
        #
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        :param s:
        :return:
        '''
        # remove all shingle from the old chunk
        self.win_shingles = {}
        # print("After Deletion S_w: ", self.S_w)
        # add all shingle from current time to the window list
        count_array = []
        for sg in s.keys():
            if len(self.win_shingles) > 0:
                if sg in self.win_shingles.keys():
                    self.win_shingles[sg].append([int(s[sg]), int(graph_count)])
                else:
                    self.win_shingles[sg] = []
                    self.win_shingles[sg].append([int(s[sg]), int(graph_count)])

            if len(self.win_shingles) == 0:
                self.win_shingles[sg] = []
                self.win_shingles[sg].append([int(s[sg]), int(graph_count)])

    def update_one_step_forward_window(self, s, graph_count, param_w):
        '''
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        # NAME: update_shingles_frequency()
        #
        # INPUTS: (s) Shingle List with count of instances for current window
        #
        # RETURN: ()
        #
        # PURPOSE: Maintain the list of shingle and their frequency in the window
        #
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        :param s:
        :return:
        '''
        # remove all shingle from the oldest window

        for id in list(self.win_shingles.keys()):
            # Find Frequency and window number
            filters = list(
                filter(lambda x: param_w <= graph_count - x[1],
                       self.win_shingles[id]))
            # Remove frequency entry from selected filters
            self.win_shingles[id] = [x for x in self.win_shingles[id] if x not in filters]

            if len(self.win_shingles[id]) == 0:  # if it has only subgraph but no frequency
                del self.win_shingles[id]

        # print("After Deletion S_w: ", self.S_w)
        # add all shingle from current time to the window list
        count_array = []
        for sg in s.keys():
            if len(self.win_shingles) > 0:
                if sg in self.win_shingles.keys():
                    self.win_shingles[sg].append([int(s[sg]), int(graph_count)])
                else:
                    self.win_shingles[sg] = []
                    self.win_shingles[sg].append([int(s[sg]), int(graph_count)])

            if len(self.win_shingles) == 0:
                self.win_shingles[sg] = []
                self.win_shingles[sg].append([int(s[sg]), int(graph_count)])

    def get_graph_sketch(self, shingles, disc_shingles):
        vec = []
        for disc_shingle in disc_shingles:
            if disc_shingle in shingles:
                vec.append(shingles[disc_shingle])
            else:
                vec.append(0)
        return vec

    def get_win_sketch(self, disc_shingles):
        self.win_sketch = []
        for disc_shingle in disc_shingles:
            if disc_shingle in self.win_shingles:
                w_count = sum(g_count[0] for g_count in self.win_shingles[disc_shingle])
                self.win_sketch.append(w_count)
            else:
                self.win_sketch.append(0)

    def get_disc_shingles(self, sketch_size):
        sh_freq = {}
        total = self.get_win_total()
        for s in self.win_shingles.keys():
            s_count = sum(g_count[0] for g_count in self.win_shingles[s])
            sh_freq[s] = s_count/total
            #sh_freq[s] = s_count

        sorted_sh = sorted(sh_freq.items(), key=lambda kv: kv[1], reverse=True)[:sketch_size]
        # print("\n\nSorted Shingle: ", sorted_sh)
        disc_shingles = []
        for sh, val in sorted_sh:
            disc_shingles.append(sh)
        #print("\n\nDiscriminative Shingles: ", disc_shingles)
        return disc_shingles

    def shingle_sketch(self, graphs, args, is_gexf):
        param_w = args.win_size
        jaccard = []
        cosine = []
        index = 0
        sketch_vecs = []
        #for g in graphs:
        for g in tqdm(range(1, len(graphs)+1)):
            if is_gexf:
                graph = graphs[g]
            else:
                graph = graph_utils.create_graph(graphs[g])

            walk_len = len(graph.edges())*args.N
            # print("Edge Count: ", walk_len)

            walk_path = self.random_walk(graph, walk_len)
            shingles = self.generate_shingles(walk_path, walk_len, args.k_shingle)
            # graph_utils.draw_graph(graph, g)

            disc_shingles = self.get_disc_shingles(args.sketch_size)

            # self.get_win_sketch(disc_shingles) # not in use now
            #print("\n\n Window Sketech: ", self.win_sketch)

            sketch_vec = self.get_graph_sketch(shingles, disc_shingles)
            sketch_vecs.append(sketch_vec)
            #print("\n\n Graph Sketech: ", sketch_vec)
            # print(disc_shingles)

            if index >= param_w:
                jaccard.append(self.calculate_similarity(shingles))

                # cosine.append(spatial.distance.cosine(self.win_sketch, sketch_vec))

            self.update_one_step_forward_window(shingles, index, param_w)

            index += 1

        sketch_vecs = np.array(sketch_vecs[3:]).astype(np.float64)
        # print("Vector : \n", sketch_vecs)

        np.savetxt(args.sketch_vector, sketch_vecs, delimiter=',')

        # print("Coefficient: ", cosine)
        # i = param_w
        # Anomalies = []
        # for j in cosine:
        #     if j < 0.002:
        #         print("Anomalies: ", i, " --- ", j)
        #         Anomalies.append(i)
        #     i += 1
        #print("Anomalies: ", Anomalies)

        # pyplot.plot(jaccard[param_w:], marker='.')
        # pyplot.savefig("jaccard.pdf")

        # pyplot.plot(cosine, marker='.')
        # pyplot.savefig("cosine.pdf")
        # pyplot.show()
