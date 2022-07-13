import os
from pathlib import Path
from dotenv import load_dotenv
from process_data import apply_SAB_preprocess
from umls_api.bioportal_api import BioPortalAPI

def get_BIOPORTAL_API_KEY():
    # Import the .env file
    dotenv_path = Path('.env')
    load_dotenv(dotenv_path=dotenv_path)

    try:
        BIOPORTAL_API_KEY = os.getenv('BIOPORTAL_API_KEY')
        return BIOPORTAL_API_KEY
    except Exception as e:
        print("Env file not good")
        return None

class Node:
    """Class for the Node."""

    def __init__(self, features):
        """Initialize the Node class."""
        self.label = features['pref_label']
        self.labels = features['labels']
        self.source = features['source']
        self.code_id = features['code_id']
        self.predicted = None # T127
        self.is_good_prediction = None # None | True | False
        self.parents_code_id = features['parents_code_id']
        self.next = []
        self.previous = []
        self.has_definition = features['has_definition']
        self.definition = features['definition']
        self.parents_type = features['parents_type']

    def add_child(self, child):
        self.next.append(child)

    def add_parent(self, parent):
        self.previous.append(parent)


class LinkedTree:
    """Class for the LinkedTree."""

    def __init__(self, initial_node):
        """Initialize the LinkedTree class."""
        self.nodes = [initial_node]

    def add_node(self, parent_code_id, new_node: Node):
        for node in self.nodes:
            if node.code_id == parent_code_id:
                new_node.parents_type = node.parents_type
                new_node.add_parent(node)
                node.add_child(new_node)
                self.nodes.append(new_node)
                return True
            
    def get_node_from_code_id(self, code_id):
        for node in self.nodes:
            if node.code_id == code_id:
                return node
        return None

    def display_node_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout

        G = nx.Graph()
        G.add_node(self.nodes[0])
        self.recursively_add_edges(self.nodes[0], G)

        color_map = []
        labels_map = {}
        for node in G.nodes():
            # Coloring
            if node.is_good_prediction is True:
                color_map.append('green')
            elif node.is_good_prediction is False:
                color_map.append('red')
            else:
                color_map.append('blue')
                
            # Labeling
            labels_map[node] = node.label

        options = {
            "font_size": 10,
            "alpha": 0.7,
            "node_color": color_map,
            "labels": labels_map
        }

        pos = graphviz_layout(G, prog="dot")
        nx.draw_networkx(G, pos, **options)

        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

    def recursively_add_edges(self, node, G):
        for child in node.next:
            parent_node = self.get_node_from_code_id(child.parents_code_id)
            G.add_edge(child, parent_node)
            self.recursively_add_edges(child, G)

def set_graph_prediction():
    answer = input("Do you want to test model graph prediction? (y/n) ")
    if answer == "y" or answer == "Y" or answer == "":
        answer = input("Enter the name of the concept (default=Melanoma): ")
        name = answer if answer != "" else "Melanoma"
        answer = input("Enter the source of the concept (default=MedDRA): ")
        source = answer if answer != "" else "MedDRA"
        answer = input("Enter the depth of the graph (default=3): ")
        try: depth = int(answer) if answer != "" else 3
        except: depth = 3
        
        BIOPORTAL_API_KEY = get_BIOPORTAL_API_KEY()
        if BIOPORTAL_API_KEY is None:
            print("BIOPORTAL_API_KEY is not set.")
            return None
        portal = BioPortalAPI(api_key=BIOPORTAL_API_KEY)

        print("Loading concepts...")
        root_link = portal.get_root_of_tree(name, source)
        features = portal.get_features_from_link(root_link, None)
        new_node = Node(features)
        linked_tree = LinkedTree(new_node)
        children_link = features['children']
        parents_code_id = features['code_id']
        children_links_list = portal.get_children_links(children_link)

        def recursive_add_all_nodes(portal, children_links_list, parents_code_id, max_depth):
            if max_depth == 0:
                return
            for link in children_links_list:
                features = portal.get_features_from_link(link, parents_code_id)
                if features is None:
                    continue
                features['source'] = apply_SAB_preprocess(features['source'])
                print(features['source'])
                new_node = Node(features)
                linked_tree.add_node(parents_code_id, new_node)
                children_link = features['children']
                new_parents_code_id = features['code_id']
                children_links_list = portal.get_children_links(children_link)
                recursive_add_all_nodes(
                    portal, children_links_list, new_parents_code_id, max_depth - 1)

        recursive_add_all_nodes(portal, children_links_list, parents_code_id, depth - 1)
        print("Number of nodes", len(linked_tree.nodes))
        linked_tree.display_node_graph() # * Display when the model prediected all nodes
        return linked_tree
    
set_graph_prediction()
