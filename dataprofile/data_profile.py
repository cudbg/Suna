import networkx as nx
import numpy as np
import pandas as pd
import random
import sys
from collections import deque
from scipy.optimize import minimize
# sys.path.append('/home/sagemaker-user/CI-Kitana/CBDS')
sys.path.append('../')

class DataProfile:
    def __init__(self, debug=False, seed=10):
        self.G = None
        self.paths = None
        self.D = None
        self.edge_coeffs = None
        self.data_in = None
        self.data_corpus = None
        self.corpus_features = []
        self.join_key = None
        self.open_paths = None # open paths between treatment and outcome
        self.join_key_domain = {}
        self.noises = {}
        self.debug = debug
        random.seed(seed)
        np.random.seed(seed)

    def generate_G(self, n, density=0.25):
        G = nx.DiGraph()

        self.ordered_nodes = [f'V{i}' for i in range(n)]
        G.add_nodes_from(self.ordered_nodes)

        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    G.add_edge(f'V{i}', f'V{j}')

        self.G = G

    def farthest_ancestors(self, node):
        ancestors = nx.ancestors(self.G, node)
        farthest_ancestors = [ancestor for ancestor in ancestors if not nx.ancestors(self.G, ancestor)]

        return farthest_ancestors

    def generate_D_from_G(self, num_samples=1000, join_key=('join_key', 10000)):
        """
        Gerenate the complete dataset that represents the full population
        """
        D = pd.DataFrame(index=range(num_samples))
        edge_coeffs = {}

        # Add the join key to the data
        jk, _ = join_key
        self.join_key = jk
        self.join_key_domain[jk] = np.arange(0, num_samples)
        D[jk] = self.join_key_domain[jk]

        for node in self.ordered_nodes:
            parents = list(self.G.predecessors(node))
            if parents:
                node_coeffs = np.random.uniform(2, 3, size=len(parents))
                noise = np.random.uniform(0, 1, size=num_samples)
                self.noises[node] = noise
                D[node] = np.round(np.dot(D[parents], node_coeffs) / len(parents) + noise, 2)
                for i, parent in enumerate(parents):
                    edge_coeffs[(parent, node)] = node_coeffs[i] / len(parents)
            else:
                noise = np.random.uniform(0, 1, size=num_samples)
                self.noises[node] = noise
                D[node] = np.round(noise, 2)

        self.D = D
        self.edge_coeffs = edge_coeffs
        
    def generate_partitions_from_D(self, treatment, target, join_key):
        corpus_features = [col for col in self.D.columns if col not in [treatment, target]]
        data_in =  self.D[[treatment, target] + join_key]
        data_corpus = self.D[corpus_features]
        self.data_in = data_in
        self.data_corpus = data_corpus
        self.corpus_features = [col for col in self.D.columns if col not in [treatment, target] + join_key]
        
    def get_ground_truth(self, treatment, outcome):
        def all_paths_dag(start, end, path=None, paths=None):
            if paths is None:
                paths = []
            if path is None:
                path = []

            path = path + [start]

            if start == end:
                paths.append(path)
            else:
                for successor in self.G.successors(start):
                    # print(int(successor[1:]), int(outcome[1:]))
                    if int(successor[1:]) <= int(outcome[1:]):
                        all_paths_dag(successor, end, path, paths)

            return paths
        
        paths = all_paths_dag(treatment, outcome)
        treatment_effects = 0
        for path in paths:
            cur_path_effect = 1
            for i in range(len(path)-1):
                edge_coeff = self.edge_coeffs[(path[i], path[i+1])]
                cur_path_effect *= edge_coeff
            treatment_effects += cur_path_effect
        return treatment_effects
    
    def get_open_paths(self, treatment, outcome, adj_sets={}):
        paths = list(nx.all_simple_paths(self.G.to_undirected(), source=treatment, target=outcome))
        open_paths = []
        for path in paths:
            if self.path_is_open(path, adj_sets):
                open_paths.append(path)
        return open_paths
        
    def get_assoc_w_adj(self, treatment, outcome, adj_sets={}):
        assoc = 0
        open_paths = self.get_open_paths(treatment, outcome, adj_sets)
        for path in open_paths:
            cur_path_assoc = 1
            for i in range(len(path)-1):
                if (path[i], path[i+1]) in self.edge_coeffs:
                    cur_path_assoc *= self.edge_coeffs[(path[i], path[i+1])]
                else:
                    cur_path_assoc *= self.edge_coeffs[(path[i+1], path[i])]
            assoc += cur_path_assoc
        return assoc
    
    def get_hyp_set(self, treatment, outcome, adj_sets={}):
        G = self.G.copy()
        outgoing_edges = []
        for adj_node in adj_sets:
            cur_outgoing_edges = [(adj_node, neighbor) for neighbor in G.successors(adj_node)]
            outgoing_edges += cur_outgoing_edges
        G.remove_edges_from(outgoing_edges)
        t_ancs = nx.ancestors(G, treatment)
        bcd_nodes = []
        for anc in t_ancs:
            if self.debug:
                print(f"This is the ancestor to test: {anc}")
            if self.is_valid_Z(anc, treatment, adj_sets, check_descendant=False):
                bcd_nodes.append(anc)
        if self.debug: print(f"BCD nodes: {bcd_nodes}")
        conf_nodes = []
        for node in bcd_nodes:
            if not self.is_valid_Z(node, outcome, adj_sets):
                conf_nodes.append(node)
                continue
            else:
                paths_to_t = list(nx.all_simple_paths(G, source=node, target=treatment))
                paths_to_targ = list(nx.all_simple_paths(G, source=node, target=outcome))
                is_conf = False
                if self.debug: print(f"Paths to Treatment: {paths_to_t}, Paths to Target: {paths_to_targ}")
                for path_to_targ in paths_to_targ:
                    for path_to_t in paths_to_t:
                        if set(path_to_targ).intersection(path_to_t) == {node}:
                            conf_nodes.append(node)
                            is_conf = True
                            break
                    if is_conf: break
        return conf_nodes
    
    # High level idea: maintain triplets, instead of paths
    # Maintain a set of visited nodes, caled visited = {}
    # Starting from the treatment variable, prev_conn, get all connection nodes, cur_conn
    # so they form pairs (prev_node, cur_node)
    # add those pairs in visited
    # For each pair (prev_node, cur_node), discover a set of its connected nodes, 
    # such that (cur_node, next_node) not in visited and (prev_node, cur_node, next_node) is open
    # pairs (prev_node, cur_node) <- (cur_node, next_node)
    def is_valid_Z(self, treatment, outcome, adj_sets={}, check_descendant=True):
        invalid_set = nx.descendants(self.G, treatment)
        if check_descendant and set(invalid_set).intersection(adj_sets):
            return False
        
        def get_connection(G, node):
            parents = list(G.predecessors(node))
            children = list(G.successors(node))
            return parents, children
        
        def update_cur_trip(cur_frontier, next_nodes, path_sets, cur_node, direction):
            for node in next_nodes:
                for path in path_sets:
                    if node not in path:
                        if (cur_node, node, direction) in cur_frontier:
                            cur_frontier[(cur_node, node, direction)].add(path + (node, ))
                        else:
                            cur_frontier[(cur_node, node, direction)] = {path + (node, )}
        
        G = self.G.copy()
        outgoing_edges = [(treatment, neighbor) for neighbor in G.successors(treatment)]
        G.remove_edges_from(outgoing_edges)
        
        treat_par, treat_chi = get_connection(G, treatment)
        # Use 0 for <- and 1 for ->
        # We store the last directed edge and all open simple paths that ends at this last directed edge
        # keep track of the triplets (Vi, Vj, direction, current paths)
        cur_frontier = {}
        for node in treat_par:
            cur_frontier[(treatment, node, 0)] = {(treatment, node)}
        for node in treat_chi:
            cur_frontier[(treatment, node, 1)] = {(treatment, node)}

        while cur_frontier:
            prev_node, cur_node, direction = list(cur_frontier.keys())[0]
            path_sets = cur_frontier[(prev_node, cur_node, direction)]
            cur_par, cur_chi = get_connection(G, cur_node)
            del cur_frontier[(prev_node, cur_node, direction)]
            
            if direction == 0 and cur_node not in adj_sets:
                if (outcome in cur_par) or (outcome in cur_chi):
                    if self.debug: print(f"A confounding path is: {list(path_sets)[0] + (outcome, )}")
                    return False
                update_cur_trip(cur_frontier, cur_par, path_sets, cur_node, 0)
                update_cur_trip(cur_frontier, cur_chi, path_sets, cur_node, 1)
            
            elif direction == 1 and cur_node in adj_sets:
                if outcome in cur_par:
                    if self.debug: print(f"A confounding path is: {list(path_sets)[0] + (outcome, )}")
                    return False
                update_cur_trip(cur_frontier, cur_par, path_sets, cur_node, 0)
                
            elif direction == 1 and cur_node not in adj_sets:
                if outcome in cur_chi:
                    if self.debug: print(f"A confounding path is: {list(path_sets)[0] + (outcome, )}")
                    return False
                update_cur_trip(cur_frontier, cur_chi, path_sets, cur_node, 1)
            if self.debug:
                print(f"This is the current frontier: {cur_frontier}")

        return True
    
    def is_collider(self, triple):
        return self.G.has_edge(triple[0], triple[1]) and self.G.has_edge(triple[2], triple[1])
    
    def path_is_open(self, path, adj_sets={}):
        if len(path) < 3: return True
        for i in range(len(path) - 2):
            triple = path[i:i+3]
            collider = self.is_collider(triple)
            middle_node_in_adj = triple[1] in adj_sets

            # Rule 1: Non-collider and middle node in adj_sets
            if not collider and middle_node_in_adj:
                return False

            # Rule 2: Collider and middle node not in adj_sets
            if collider and not middle_node_in_adj:
                return False

        return True