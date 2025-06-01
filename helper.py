import networkx as nx
import pandas as pd
import random
import math
import numpy as np
import os.path
import Graph_Sampling
import time
import psutil

def graph_glimpse(graph):
    """glimpse a graph_object"""
    #print("Type of graph: ", type(graph))
    print("Number of nodes: ", len(graph.nodes()))
    print("Number of edges: ", len(graph.edges()))
    print("Some first nodes: ", list(graph.nodes())[0:100])
    print("Some first edges: ", list(graph.edges())[0:100])

def match_new_mtx_index_dat(sample_graph):
    """
    IN:
    sample_graph: an output of any sub-sampling function, it carries all (sub-sampled) nodes and egde with "mtx_index" attribute,
                the "mtx_index" is the original index gotten from the original_graph that the sample_graph gotten from 
    OUT:
    a pandas data frame with two columns "new_index" and mtx_index
    """
    sample_attribute = nx.get_node_attributes(sample_graph, "mtx_index")
    sample_attribute = pd.DataFrame(sample_attribute, index =[0])
    sample_attribute = sample_attribute.transpose().reset_index().rename(columns = {"index": "new_index", 0: "mtx_index" })
    return(sample_attribute)
####################################################################################################
## For applying original sub-sampling method without any alteration 
####################################################################################################
def create_sampler_and_do_sampling(graph, n= 100, sub_method = "SRW"):
    """A function to create subsampling-algorithm-object and do the sub-sampling
    This function will be used inside the "combine_sub_sampling" function 
    IN: 
    graph: a (original) graph created from a mtx file
    n: number of nodes that we want to subsample
    sub_method: (string), the method to do subsampling, can be   "RNS", "SRW", "SRWFB", "FF", "MH", "SB", ...
    OUT: 
    sub_graph: a sub_graph having n nodes from graph using sub_method subsampling method"""
    #random.seed(6357)
    match sub_method:
        case "SRW":
            object = Graph_Sampling.SRW_RWF_ISRW()
            sub_graph = object.random_walk_sampling_simple(graph,n) # graph, number of nodes to sample
        case "InSRW":
            object = Graph_Sampling.SRW_RWF_ISRW()
            sub_graph = object.random_walk_induced_graph_sampling(graph,n)
        case "SRWFB":
            object = Graph_Sampling.SRW_RWF_ISRW()
            sub_graph = object.random_walk_sampling_with_fly_back(graph, n, 0.15) # graph, number of nodes to sample, p(fly back)
        case "InSRWFB":
            object = Graph_Sampling.SRW_RWF_ISRW()
            sub_graph = object.induced_random_walk_sampling_with_fly_back(graph, n, 0.15)
        case "MH": ## Metropolis Hasting
            object = Graph_Sampling.MHRW()
            parent_node = random.sample(list(graph.nodes()), 1)[0]
            sub_graph = object.mhrw(graph, parent_node, n)
        case "InMH":
            object = Graph_Sampling.MHRW()
            parent_node = random.sample(list(graph.nodes()), 1)[0]
            sub_graph = object.induced_mhrw(graph, parent_node, n)
        case "SB": ##Snowball
            object = Graph_Sampling.Snowball()
            sub_graph = object.snowball(graph, n,100)
        case "PR": ## PageRank
            from littleballoffur import PageRankBasedSampler
            object = PageRankBasedSampler(number_of_nodes = n, alpha = 0.85)
            graph_renumbered = nx.convert_node_labels_to_integers(graph, label_attribute='mtx_index')
            sub_graph = object.sample(graph_renumbered)
            mapping = {node: data['mtx_index'] for node, data in sub_graph.nodes(data=True)}
            sub_graph = nx.relabel_nodes(sub_graph, mapping)
        case "InPR": ##"Induce PageRank"
            from littleballoffur import PageRankBasedSampler
            object = PageRankBasedSampler(number_of_nodes = n, alpha = 0.85)
            graph_renumbered = nx.convert_node_labels_to_integers(graph, label_attribute='mtx_index')
            sub_graph = object.sample(graph_renumbered)
            sub_graph_node = sub_graph.nodes()
            sub_graph = graph_renumbered.subgraph(sub_graph_node)
            mapping = {node: data['mtx_index'] for node, data in sub_graph.nodes(data=True)}
            sub_graph = nx.relabel_nodes(sub_graph, mapping)
        case "RNS": ## Random Node Sampling
            all_nodes = list(graph.nodes())
            sampled_nodes = random.sample(all_nodes, n)
            sub_graph = nx.Graph()
            for node in sampled_nodes:
                sub_graph.add_node(node)
                sub_graph.add_edge(node, node)
        case "InRNS": ## Random Node Sampling
            all_nodes = list(graph.nodes())
            sampled_nodes = random.sample(all_nodes, n)
            sub_graph = graph.subgraph(sampled_nodes) 
        case unknown_command:
            print("subs-sampling method is unknown")
    return(sub_graph)
## -----------------------------------------------------------------------
def do_sampling_for_a_sample_percent_list(graph, sample_size_percent_l, percent_indi = True, 
                                          sub_method = "MH", 
                                          output_dir = "./", 
                                          induce=False):
    """
    sample_size_percent_l (vector): vector of sample_size or percentage_of_number_of nodes taht we want to subsample
    percent_indi (boolean): if True: do sampling from a vector percentage of total number of nodes from original graph ; 
                            if False: do sampling from a vector of of sample_sizes
    sub_method (string): subsampling method, can be "SRWFB", "InSRWFB", "MH", "InMH", "SB", "InSB", ...
    induce (boolean): if True, the subsampling method applied is a Induced-subsampling method, this option affects the output txt's file name
    """
    i=0
    org_net_total_num_node = len(graph.nodes())
    times = []
    memories = []
    for n in sample_size_percent_l:
        
        if percent_indi:
            m = n
            n = math.ceil((n/100)*org_net_total_num_node)  ##number of nodes we want to sample
        print("sub-graph's size: ", n)
        
        start_time = time.time()  # 
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024) 
        # make an object and call function to do subsampling
        sample = create_sampler_and_do_sampling(graph, n, sub_method)   ## do the sampling
        #sample_edges = list(sample.edges())
        #sample_nodes = list(sample.nodes())
        num_nodes = sample.number_of_nodes()
        print(f"Number of nodes: {num_nodes}")
        #sample_edges_dat = pd.DataFrame(sample_edges, columns=['from', 'to'])
        #sample_edges_dat = pd.DataFrame(sample_edges)
        ##
        if sub_method in ["SRW", "InSRW","SRWFB","InSRWFB","MH","InMH","PR", "InPR"]: 
            sample_edges_dat = get_sub_sampled_with_mtx_index(sample)
        else: 
            sample_edges_dat = get_sub_sample_edge_to_dat(sample)
       
        i += 5 # the percentage of node that we sample, each time increase by 5%
        end_time = time.time()  # 
        end_mem = process.memory_info().rss / (1024 * 1024)  #
    
        times.append(end_time - start_time)
        memories.append(end_mem - start_mem)
        if induce:
            np.savetxt(os.path.join(output_dir,"In_sub_"+ str(m)+".txt"), sample_edges_dat.values, fmt='%d')
            np.savetxt(os.path.join(output_dir, f"In_sub_{str(m)}_time.txt"), [end_time - start_time], fmt='%f')
            np.savetxt(os.path.join(output_dir, f"In_sub_{str(m)}_mem.txt"), [end_mem - start_mem], fmt='%f')
            #sample_edges_dat.to_csv(os.path.join(output_dir,"In_sub_"+ str(i)+".txt"), sep='\t', index=False, header=False)
        else:
            np.savetxt(os.path.join(output_dir,"sub_"+ str(m)+".txt"), sample_edges_dat.values, fmt='%d')
            np.savetxt(os.path.join(output_dir, f"sub_{str(m)}_time.txt"), [end_time - start_time], fmt='%f')
            np.savetxt(os.path.join(output_dir, f"sub_{str(m)}_mem.txt"), [end_mem - start_mem], fmt='%f')
            #sample_edges_dat.to_csv(os.path.join(output_dir,"sub_"+ str(i)+".txt"), sep='\t', index=False,header=False)
    return(sample_edges_dat)

##------------------------------------------------------------------------------------------
# def do_induce_for_sub_graph(org_graph, sub_graph):
#     """do the inducing of sub_graph from the original_graph (org_graph)"""

#--------------------------------------------------------------------------------------------
def get_sub_sample_edge_to_dat(sub_graph):
    """Get egdes of a sub-sample graph, and convert it to a data frame with "from" and "to" """
    sub_graph_edges_dat = sub_graph.edges()
    sub_graph_edges_dat = pd.DataFrame(sub_graph_edges_dat).rename(columns = {0: "from", 1: "to" })
    ## 
    sub_graph_edges_dat['from'] = sub_graph_edges_dat['from'].astype(str).astype(int)
    sub_graph_edges_dat['to'] = sub_graph_edges_dat['to'].astype(str).astype(int)
    return(sub_graph_edges_dat)
#--------------------------------------------------------------------------------------------
def get_sub_sampled_with_mtx_index(sample):
    """NOTE: only call this fucntion when using SRW, or SRWFB"""
    """A function to get subsample with indexes taken from the mtx file that makes the original graph.
    Since when we do sampling, we need to reset the index of the subsample to 0-sample_size, 
    This function will help return sample_edges_dat "table" for the subsample with indexes taken from the mtx file 
    so that we can know which nodes from the original_graph were sampled """
    sample_node_mtx_index = nx.get_node_attributes(sample, "mtx_index")
    ## get the subsampled-edges data 
    sample_edges_dat = sample.edges()
    sample_edges_dat = pd.DataFrame(sample_edges_dat).rename(columns = {0: "from", 1: "to" })
    ## Replace the new_subsampled_index by original index
    sample_edges_dat.replace({"from": sample_node_mtx_index}, inplace=True)
    sample_edges_dat.replace({"to": sample_node_mtx_index}, inplace=True)
    ## Note that when we run networkx, the mtx is reset to start from 0, while it actually starts from 1, 
    # we need to increase 1 for each index
    sample_edges_dat['from'] = sample_edges_dat['from'].astype(str).astype(int)
    sample_edges_dat['to'] = sample_edges_dat['to'].astype(str).astype(int)
    ##
    sample_edges_dat['from'] = sample_edges_dat['from']+1
    sample_edges_dat['to'] = sample_edges_dat['to']+1
    return(sample_edges_dat)
####################################################################################################
## For doing Combine sub-sampling method
####################################################################################################
def combine_sub_sampling(input_directory, isolated_subgraph_filename = "isolated_subgraph_dat.txt", 
                      non_isolated_subgraph_file_name = "non_isolated_subgraph_dat.txt", 
                      n_nodes_to_sample =100, iso_percent = 10, sub_method = "SRW"):
    """
    A function to do subsampling with isolated and non-isolated nodes separately and then combine them into a final subsampled_graph sample 
    IN: 
    input_directory: directory to the folder keeping isolated_subgraph_filename and non_isolated_subgraph_file_name
    n_nodes_to_sample (int): total node to sub-sample
    iso_percent (int): percentage of isolated nodes that we want to keep in the sub-sampled sample
    sub_method (string): method to do the subsampling, can be:  "RNS", "SRW", "SRWFB", "FF", "MH", "SB", ...
        RNS: Random Node Sampling
        SRW: Simple Random Walk Sampling; InSRW: Induced Subgraph Random Walk Sampling 
        SB: Snow_Ball
        FF: Forest_Fire
        MH: Metropolis Hasting; InMH: Induced Metropolis Hasting
    OUT: 
    subsampled_graph: a subsampled-graph data table with columns "from" and "to", contains all loops and edges in the subsampled graph, 
                    the node_index in the subsampled_graph is not reseted and be taken from the node_index from the mtx file that creates the original graph
    """
    #random.seed(6357)
    n1 = math.ceil((iso_percent/100)*n_nodes_to_sample)   ## Number of isolated nodes will sample
    n2 = n_nodes_to_sample - n1                           ## Number of non_isolated nodes will sample
    ##----------------------------
    ## Step 1: Create a data table of subsampled-graph containing loops created from isolated subgraph 
    ## Read the isolated_subgraph from the original_graph into an array of isolated_node_indexes 
    g1 = np.loadtxt(os.path.join(input_directory, isolated_subgraph_filename), dtype=int)
    random_iso_node = np.random.choice(g1, size=n1, replace=False, p=None)
    #
    sample1 = nx.Graph()
    for i in random_iso_node:
        sample1.add_edge(i,i)
    sample1_edge_dat = pd.DataFrame(sample1.edges()).rename(columns = {0: "from", 1: "to" })
    # print("done with sample1_edge_dat")
    ##---------------------------------
    ## Step 2: Create a data table of subsample-graph containing non-isolated node sampled using sub-sampling method (sub-method)
    # Read the non_isolated_subgraph from the original_graph
    g2= nx.read_edgelist(os.path.join(input_directory, non_isolated_subgraph_file_name), nodetype =int)  #notetype = int to be usec in Metropolis Hasting algorithm
    g2 = nx.Graph(g2)
    ## make an subsampling-method-object and do subsampling
    sample2 = create_sampler_and_do_sampling(g2, n2, sub_method)
    ## get the original index (from the original complete graph)
    if sub_method in ["SRW", "InSRW","SRWFB","InSRWFB","MH","InMH","PR", "InPR"]: 
        sample2_edges_dat = get_sub_sampled_with_mtx_index(sample2)
    else: 
        sample2_edges_dat = get_sub_sample_edge_to_dat(sample2)
    ###-----------------------------------------------------
    ## Step 3: row_bind  sample1_edge_dat and sample2_edges_dat to create sampled_graph
    sub_graph = pd.concat([sample1_edge_dat, sample2_edges_dat])
    # ## Convert all columns to int so that it will be save to txt file easily if needed
    # sub_graph['from'] = sub_graph['from'].astype(str).astype(int)
    # sub_graph['to']   = sub_graph['to'].astype(str).astype(int)k
    return(sub_graph)

def do_Combine_sampling_for_a_sample_percent_list(graph, person_input_dir, sample_size_percent_l, sub_method, iso_percent = 62, 
                                          output_dir = "D:/2023/00_Summer_intern/UCSF/NetworkSubsampling/Result/Subsampling", induce=False):
    """ do a combine sampling , can combine with normal-subsampling method or a Induced-subsampling method
    """
    org_net_total_num_node = len(graph.nodes())
    times = []
    memories = []
    for n_percent in sample_size_percent_l: 
        n = math.ceil((n_percent/100)*org_net_total_num_node)  ##number of nodes we want to sample
        print(n)
                
        start_time = time.time()  # 
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024) 
        
        # make an object and call function to do subsampling
        sample = combine_sub_sampling(input_directory= person_input_dir, 
                      n_nodes_to_sample = n, iso_percent = iso_percent, sub_method = sub_method)   ## do the sampling 
        
        end_time = time.time()  # 
        end_mem = process.memory_info().rss / (1024 * 1024)  #
    
        times.append(end_time - start_time)
        memories.append(end_mem - start_mem)
        
        if induce:
            np.savetxt(os.path.join(output_dir,"Comb_In_"+ str(n_percent)+".txt"), sample.values, fmt='%d')
            np.savetxt(os.path.join(output_dir, f"Comb_In_{str(n_percent)}_time.txt"), [end_time - start_time], fmt='%f')
            np.savetxt(os.path.join(output_dir, f"Comb_In_{str(n_percent)}_mem.txt"), [end_mem - start_mem], fmt='%f')
        else:
            np.savetxt(os.path.join(output_dir,"Comb_" + str(n_percent) +  ".txt"), sample.values, fmt='%d')
            np.savetxt(os.path.join(output_dir, f"Comb_{str(n_percent)}_time.txt"), [end_time - start_time], fmt='%f')
            np.savetxt(os.path.join(output_dir, f"Comb_{str(n_percent)}_mem.txt"), [end_mem - start_mem], fmt='%f')

# def combine_sub_sampling(input_directory, isolated_subgraph_filename = "isolated_subgraph_dat.txt", 
#                       non_isolated_subgraph_file_name = "non_isolated_subgraph_dat.txt", 
#                       n_nodes_to_sample =100, iso_percent = 10, sub_method = "SRW")
#%%#######################################################################################################
## For computing portrait divergence
##########################################################################################################
#---------------------------------------------------------------------
## function to compute portrait_divergence between the original and a subsampled networks with associated percentege of samples taken from the original network 
## portrait_divergence distance between two networks: https://github.com/bagrow/network-portrait-divergence
def com_portrait_divergence_dist(org_graph, all_sub_dir, person_id = "002-001", sub_folder = "SRWFB", sub_method = "sub", 
                                 sub_percent_l = [5, 10, 15, 20, 25, 30], write_to_xlsx = False):
    """
    person_id (string)
    sub_folder (string): the folder that contains the loops and edges for subsampled-graphs and the SubsamplingResult.xlxs file
    sub_method (string): sub-sampling method, this will be "sub" , "In_sub", "Comb", "Comb_In" 
                         corresponding with normal-subsampling, induced-subsampling, Combine-subsampling, Combine-Induce-subsampling  
    """
    portrait_divergence_dist_l = []
    for sub_percent in sub_percent_l: 
        print(sub_percent)
        if "Abundance" in person_id:
            sub_data_dir = os.path.join(all_sub_dir, person_id , sub_folder, str(sub_method)+"_"+str(sub_percent)+ "_edge" + ".txt")
        else:
            sub_data_dir = os.path.join(all_sub_dir, person_id , sub_folder, str(sub_method)+"_"+str(sub_percent)+ ".txt")
        #print(all_sub_dir)

        sub_graph = nx.read_edgelist(sub_data_dir, create_using= nx.Graph(), nodetype=int)
        portrait_divergence_dist_l.append(Graph_Sampling.portrait_divergence.portrait_divergence(org_graph, sub_graph))
    ## Create a table from sub_percent_l and portrait_divergence_dist_l
    df = pd.DataFrame({'sub_percent' : sub_percent_l, 'portrait_divergence' : portrait_divergence_dist_l})   
    ## Add the sub_method to the sub_percent column to know the method associted with the portrait_divergence
    df['sub_percent'] = sub_method + "_" + df['sub_percent'].astype(str)
    ## if write_to_xlsx==True then write the df to the  SubSamplingResult.xlsx, sheet="portrait_divergence"
    if (write_to_xlsx):
        file_path = os.path.join(all_sub_dir, person_id, sub_folder, "SubSamplingResult.xlsx")
        sheet_name = sub_folder + sub_method
        if not os.path.exists(file_path):
            mode = 'w'
        else:
            mode = 'a'  
        with pd.ExcelWriter(file_path, engine="openpyxl", mode=mode) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return(df)



