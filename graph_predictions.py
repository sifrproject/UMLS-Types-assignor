import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from process_data import get_parents_type_format

def get_nb_rows(list_of_features):
    """Get the number of rows of a list of features

    Args:
        list_of_features (List[List[Any]]): list of features

    Returns:
        int: nb_rows
    """
    return list_of_features[0].shape[0]

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
        self.has_definition = features['has_definition']
        self.definition = features['definition']
        self.parents_type = features['parents_type']
        self.parents_code_id = features['parents_code_id']
        self.prediction = None # T127
        self.next = []
        self.previous = []

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
            
    def predict_graph(self, model, dic_y_mapping, config):
        for child in self.nodes[0].next:
            self.predict_recursively(child, model, dic_y_mapping, config)

    def predict_recursively(self, node, model, dic_y_mapping, config):
        if node.prediction is None:
            print("Predicting node: " + node.label)
            features = self.get_graph_features(node, config)
            predicted_prob = model.predict(features)
            predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
            node.prediction = predicted[0]
            print("Predicted", node.prediction)
            self.update_parents_type_of_children(node)
        for child in node.next:
            self.predict_recursively(child, model, dic_y_mapping, config)
            
    def update_parents_type_of_children(self, node):
        for child in node.next:
            child.parents_type = node.prediction

    def get_graph_features(self, node, config):
        features = []
        if "Def" in config["attributes_features"]:
            features.append(np.stack([node.definition]))
        if "Has_Def" in config["attributes_features"] or "SAB" in config["attributes_features"] or \
                "Parents_Types" in config["attributes_features"]:
            features_attributes = []
            # Has_Def
            if "Has_Def" in config["attributes_features"]:
                has_def = np.stack([1 if node.has_definition is True else False])
                features_attributes.append(has_def)
            # SAB
            if "SAB" in config["attributes_features"]:
                source = np.stack(node.source)
                features_attributes.append(source)
            # Parents_Types
            if "Parents_Types" in config["attributes_features"]:
                if node.parents_type is not None:
                    parents_type = node.parents_type
                else:
                    parents_type = ""
                parents_type_format = get_parents_type_format("TUI", [parents_type])
                features_attributes.append(np.stack(parents_type_format))

            attributes = np.concatenate((features_attributes[0], features_attributes[1], features_attributes[2]), None)
            
            features.append(np.stack([attributes]))
        if "Labels" in config["attributes_features"]:
            features.append(np.stack([node.labels]))
        return features
    
    def save_prediction_to_ttl(self, source, config):
        # Save the prediction to a ttl file
        f = open("artefact/predictions_" + str(source) + ".ttl", "w")
        namespace_ontology = "@prefix onto: <https://data.bioontology.org/ontologies/> .\n"
        namespace_bpm = "@prefix sty: <http://purl.bioontology.org/ontology/STY/> .\n"
        f.write(namespace_ontology)
        f.write(namespace_bpm)
        f.write("\n")

        def recusively_write_prediction(node, f):
            if node.prediction is not None:
                f.write("onto:" + str(source) + " " + node.code_id + " sty:" + node.prediction + "\n")
            for child in node.next:
                recusively_write_prediction(child, f)

        recusively_write_prediction(self.nodes[0], f)
        f.close()

