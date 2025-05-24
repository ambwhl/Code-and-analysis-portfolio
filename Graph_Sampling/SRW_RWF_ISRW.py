import random
# import time
# import datetime
# import io
# import array, re, itertools
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
# from itertools import groupby


class SRW_RWF_ISRW:

    def __init__(self):
        self.growth_size = 2
        self.T = 100    # number of iterations
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.15
        self.G1 = nx.Graph()

    def random_walk_sampling_simple(self, complete_graph, n_nodes_to_sample):
        #----
        # Set up initial conditions
        #-----
        ## V1: don't care about the original index
        # complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True )
        ## V2: Care about the original index of the original graph
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', label_attribute='mtx_index')
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = n_nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        # add node with id = new index for sampling and original_index = original index from the original network 
        # sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])
        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'], 
                               mtx_index=complete_graph.nodes[index_of_first_random_node]['mtx_index'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node

        #---------------------
        # Run loop to sample nodes until we get n_nodes_to_sample
        #---------------------
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            ## Find neighbors of the current node, randomly choose a node in the neighbors, 
            ## Then add the node and the edge that connects the current node and the chosen node
            edges = [n for n in complete_graph.neighbors(curr_node)]  ## Find neighbors of the current node
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            ## V1: without consider teh original_index from the original graph
            ## sampled_graph.add_node(chosen_node)
            sampled_graph.add_node(chosen_node, mtx_index=complete_graph.nodes[chosen_node]['mtx_index'])
            sampled_graph.add_edge(curr_node, chosen_node)

            ## Update current node and iteration 
            curr_node = chosen_node
            iteration = iteration + 1

            # if iteration == maximum number of iteration (T), then randomly chose a new current node
            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                edges_before_t_iter = sampled_graph.number_of_edges()

        for n, data in sampled_graph.nodes(data=True):
            if 'mtx_index' not in data:
                if 'mtx_index' in complete_graph.nodes[n]:
                    mtx_index_value = complete_graph.nodes[n]['mtx_index']
                    sampled_graph.nodes[n]['mtx_index'] = mtx_index_value
                    print(f"Added 'mtx_index' to Node {n}")
                else:
                    print(f"Node {n} does not have 'mtx_index' in complete_graph either")
      
        remapped_graph = nx.relabel_nodes(sampled_graph, {node: sampled_graph.nodes[node]['mtx_index'] for node in sampled_graph.nodes()})
        return remapped_graph

    def random_walk_sampling_with_fly_back(self, complete_graph, n_nodes_to_sample, fly_back_prob):
        ## V1: don't care about the original index
        # complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True )
        ## V2: Care about the original index of the original graph
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', label_attribute='mtx_index')
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n
            if 'mtx_index' not in data:
                print(f"Node {n} does not have 'mtx_index'")

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = n_nodes_to_sample

        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        ##sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])
        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'], 
                               mtx_index = complete_graph.nodes[index_of_first_random_node]['mtx_index'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            # sampled_graph.add_node(chosen_node)
            #sampled_graph.add_node(chosen_node, mtx_index = complete_graph.nodes[chosen_node]['mtx_index'])
            sampled_graph.add_node(complete_graph.nodes[chosen_node]['id'], mtx_index = complete_graph.nodes[chosen_node]['mtx_index'])
            sampled_graph.add_edge(curr_node, chosen_node)
            choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                    #print("Choosing another random node to continue random walk ")
                edges_before_t_iter = sampled_graph.number_of_edges()
        print("upper_bound_nr_nodes_to_sample sampling done ")

        for n, data in sampled_graph.nodes(data=True):
            if 'mtx_index' not in data:
                if 'mtx_index' in complete_graph.nodes[n]:
                    mtx_index_value = complete_graph.nodes[n]['mtx_index']
                    sampled_graph.nodes[n]['mtx_index'] = mtx_index_value
                    print(f"Added 'mtx_index' to Node {n}")
                else:
                    print(f"Node {n} does not have 'mtx_index' in complete_graph either")
      
        remapped_graph = nx.relabel_nodes(sampled_graph, {node: sampled_graph.nodes[node]['mtx_index'] for node in sampled_graph.nodes()})
        return remapped_graph

    def random_walk_induced_graph_sampling(self, complete_graph, n_nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', label_attribute='mtx_index')
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = n_nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)

        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])

        iteration = 1
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        sampled_graph = complete_graph.subgraph(Sampled_nodes)
        remapped_graph = nx.relabel_nodes(sampled_graph, {node: sampled_graph.nodes[node]['mtx_index'] for node in sampled_graph.nodes()})
        return remapped_graph
        #return sampled_graph
    
    def induced_random_walk_sampling_with_fly_back(self, complete_graph, n_nodes_to_sample, fly_back_prob):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', label_attribute='mtx_index')
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = n_nodes_to_sample

        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        ##sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])
        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'], 
                               mtx_index = complete_graph.nodes[index_of_first_random_node]['mtx_index'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            # sampled_graph.add_node(chosen_node)
            sampled_graph.add_node(chosen_node, mtx_index = complete_graph.nodes[chosen_node]['mtx_index'])
            sampled_graph.add_edge(curr_node, chosen_node)
            choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                    #print("Choosing another random node to continue random walk ")
                edges_before_t_iter = sampled_graph.number_of_edges()
        
        induced_graph = complete_graph.subgraph(sampled_graph.nodes())
        for n, data in induced_graph.nodes(data=True):
            if 'mtx_index' not in data:
                if 'mtx_index' in complete_graph.nodes[n]:
                    mtx_index_value = complete_graph.nodes[n]['mtx_index']
                    induced_graph.nodes[n]['mtx_index'] = mtx_index_value
                    print(f"Added 'mtx_index' to Node {n}")
                else:
                    print(f"Node {n} does not have 'mtx_index' in complete_graph either")
        remapped_graph = nx.relabel_nodes(induced_graph, {node: induced_graph.nodes[node]['mtx_index'] for node in induced_graph.nodes()})
        return remapped_graph
        #return sampled_graph
