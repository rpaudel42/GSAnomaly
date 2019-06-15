import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd


def get_gex_graph(filename):
    G = nx.read_gexf(filename)
    return G

def create_graph(graph):
    G = nx.Graph()
    node ={}
    for n in graph['node']:
        node[n] = graph['node'][n]
        G.add_node(n, label = graph['node'][n])
    for e in graph['edge']:
        src, dest = e.split(' ')
        G.add_edge(src, dest, label = graph['edge'][e])
    return G

def read_graph(file_name):
    '''
    PURPOSE: Read Synthetic Graph and load as dictionary of a graph file
    :param file_name:
    :param label:
    :return:
    '''
    graph = {}
    node = {}
    edge = {}
    with open(file_name) as f:
        lines = f.readlines()
    for l in lines:
        graph_entry = l.split(" ")
        if graph_entry[0] == 'v':
            node[graph_entry[1]] = graph_entry[2].strip('\n')
        elif graph_entry[0] == 'd' or graph_entry[0] == 'u':
            edge[graph_entry[1] + ' ' + graph_entry[2].strip('\n')] = graph_entry[3].strip("\n")
    graph["node"] = node
    graph["edge"] = edge
    graph["label"] = "P"
    return graph

def read_send_gfiles(fileName):

    '''PURPOSE: Read .G Graph, load each XP as JSON and send each XP as a graph stream
            :param fileName:
            :return:
    '''
    graph = {}
    node = {}
    edge = {}
    g_list = {}
    XP = 0
    label = 'pos'
    with open(fileName) as f:
        lines = f.readlines()
        for line in lines:
            singles = line.split(' ')
            if singles[0] == "XP" or singles[0] == "XN":
                if singles[0] == "XP":
                    label = 'pos'
                if singles[0] == "XN":
                    label = 'neg'
                if XP > 0:
                    graph["node"] = node
                    graph["edge"] = edge
                    graph["label"] = label
                    g_list[XP] = graph

                graph = {}
                node = {}
                edge = {}
                XP += 1
            elif singles[0] == "v":
                node[singles[1]] = singles[2].strip('\n').strip('\"')
            elif (singles[0] == "u" or singles[0] == "d"):
                edge[singles[1] + ' ' + singles[2]] = singles[3].strip('\n').strip('\"')
    return g_list

def get_graph(filename):
    G=nx.Graph()
    f=open(filename,'r')
    lines=f.readlines()
    for line in lines:
        if(line[0]=='#'):
            continue
        else:
            temp=line.split()
            index1=int(temp[0])
            index2=int(temp[1])
            G.add_edge(index1,index2)         
    f.close()
    return G

def create_graphs(csv_file, graph_folder):
    print("\n\n ---- Creating G Files -----")
    tcp = pd.DataFrame(index=[], columns=[])
    tcp1 = pd.read_csv(csv_file)
    tcp = tcp.append(tcp1, ignore_index=True)
    tcp = tcp.iloc[:, [1, 2, 3, 4]]
    tcp.columns = ['source', 'destination', 'anomaly','hours_past']

    global_nodes = {}
    local_node = {}
    global_node_id = 1
    local_node_id = 1
    hour = 0
    is_new_graph = False
    fw = open(graph_folder + "/" + str(hour) + ".g", "w")
    fw.write("XP # 1\n")
    for index, row in tcp.iterrows():
        if 1==1: #row['hours_past'] < 3:
            if row['source'] not in global_nodes:
                global_nodes[row['source']] = global_node_id
                global_node_id += 1

            if row['destination'] not in global_nodes:
                global_nodes[row['destination']] = global_node_id
                global_node_id += 1

            curr_hour = row['hours_past']

            if hour != curr_hour:
                print("\n\n Hour Past: ", hour, "    ", index)
                fw = open(graph_folder + "/" + str(curr_hour) + ".g", "w")
                fw.write("XP # 1\n")
                local_node =  {}
                local_node_id = 1
            else:
                fw = open(graph_folder + "/" + str(curr_hour) + ".g", "a")

            hour = row['hours_past']

            if row['source'] not in local_node:
                local_node[row['source']] = local_node_id #global_nodes[row['source']]
                fw.write("v " + str(local_node_id) + " \"" + str(global_nodes[row['source']]) + "\"\n")
                local_node_id += 1

            if row['destination'] not in local_node:
                local_node[row['destination']] = local_node_id #global_nodes[row['destination']]
                fw.write("v " + str(local_node_id) + " \"" + str(global_nodes[row['destination']]) + "\"\n")
                local_node_id += 1

            fw.write("d " + str(local_node[row['source']]) + " " + str(local_node[row['destination']]) + " \"call\"\n")


def random_walk(G, walkSize):
    walkList = []
    node_ids = []
    #print("G: ", G.nodes(), G.edges())
    for n in G.nodes(data=True):
        node_ids.append(n)
        curNode = n[0]
        # curNode = random.choice(node_ids)
        # curNode = curNode[0]
        while (len(walkList) < walkSize):
            walkList.append(curNode)
            try:
                curNode = random.choice(list(G.neighbors(curNode)))
            except:
                print("Cur Node: ", curNode, "Neighbor: ", list(G.neighbors(curNode)))
                break

    return walkList


def random_walk_sub2_vec(G, walkSize):
    walkList= []
    node_ids = []
    for n in G.nodes(data=True):
        node_ids.append(n)
    curNode = random.choice(node_ids)
    curNode = curNode[0]
    while(len(walkList) < walkSize):
        walkList.append(curNode)
        try:
            curNode = random.choice(list(G.neighbors(curNode)))
        except:
            print("Cur Node: ", curNode, "Neighbor: ", list(G.neighbors(curNode)))
            break
    return walkList
    
def get_stats(G):
    stats ={}
    stats['num_nodes'] = nx.number_of_nodes(G)
    stats['num_edges'] = nx.number_of_edges(G)
    stats['is_Connected'] = nx.is_connected(G)


def draw_graph(G, index):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.savefig("data/pdfs/"+str(index)+".pdf")
    #plt.show()
