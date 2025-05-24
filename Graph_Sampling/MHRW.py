import random
import networkx as nx

class MHRW():
    def __init__(self):
        self.G1 = nx.Graph()

    def mhrw(self, G, node, size):
        G = nx.convert_node_labels_to_integers(G, 0, 'default', label_attribute='mtx_index')
        for n, data in G.nodes(data=True):
            G.nodes[n]['id'] = n

        node = random.choice(list(G.nodes()))

        node_list = set()
        dictt = {}

        node_list.add(node)
        parent_node = node
        dictt[parent_node] = parent_node

        for _ in range(size - 1):
            if parent_node not in G:
                print(f"parent_node {parent_node} not in graph. Skipping step.")
                break

            neighbors = list(G.neighbors(parent_node))
            if not neighbors:
                print(f"parent_node {parent_node} has no neighbors. Skipping step.")
                break

            next_node = random.choice(neighbors)
            degree_parent = G.degree(parent_node)
            degree_next = G.degree(next_node)

            acceptance_ratio = degree_parent / degree_next if degree_next != 0 else 0

            if random.random() < acceptance_ratio:
                if next_node not in dictt:
                    node_list.add(next_node)
                    dictt[next_node] = next_node
                parent_node = next_node

        return G.subgraph(node_list)

    def induced_mhrw(self, G, node, size):
        G = nx.convert_node_labels_to_integers(G, 0, 'default', label_attribute='mtx_index')
        for n, data in G.nodes(data=True):
            G.nodes[n]['id'] = n

        node = random.choice(list(G.nodes()))

        node_list = set()
        dictt = {}

        node_list.add(node)
        parent_node = node
        dictt[parent_node] = parent_node

        for _ in range(size - 1):
            if parent_node not in G:
                print(f"parent_node {parent_node} not in graph. Skipping step.")
                break

            neighbors = list(G.neighbors(parent_node))
            if not neighbors:
                print(f"parent_node {parent_node} has no neighbors. Skipping step.")
                break

            next_node = random.choice(neighbors)
            degree_parent = G.degree(parent_node)
            degree_next = G.degree(next_node)

            acceptance_ratio = degree_parent / degree_next if degree_next != 0 else 0

            if random.random() < acceptance_ratio:
                if next_node not in dictt:
                    node_list.add(next_node)
                    dictt[next_node] = next_node
                parent_node = next_node

        return G.subgraph(node_list)
