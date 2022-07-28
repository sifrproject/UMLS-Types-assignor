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


def split_code_id(code_id):
    # Reverse the code_id
    code_id = code_id[::-1]
    arr = code_id.split("/")
    code = arr[0][::-1]
    link = code_id[len(code):][::-1]
    return code, link


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
        self.prediction = None  # T127
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
            predicted = [dic_y_mapping[np.argmax(
                pred)] for pred in predicted_prob]
            node.prediction = predicted[0]
            print("Predicted", node.prediction)
            self.update_parents_type_of_children(node)
        for child in node.next:
            self.predict_recursively(child, model, dic_y_mapping, config)

    def recursively_find_all_children(self, node, code_id, list_children):
        if node.code_id == code_id:
            list_children.append(node)
        for child in node.next:
            self.recursively_find_all_children(child, code_id, list_children)
        return list_children

    def update_parents_type_of_children(self, node):
        list_children = self.recursively_find_all_children(
            self.nodes[0], node.code_id, [])
        for child in list_children:
            for parents_type in node.parents_type:
                child.parents_type.append(parents_type)
            

    def get_graph_features(self, node, config):
        features = []
        if "Def" in config["attributes_features"]:
            features.append(np.stack([node.definition]))
        if "Has_Def" in config["attributes_features"] or "SAB" in config["attributes_features"] or \
                "Parents_Types" in config["attributes_features"]:
            features_attributes = []
            # Has_Def
            if "Has_Def" in config["attributes_features"]:
                has_def = np.stack(
                    [1 if node.has_definition is True else False])
                features_attributes.append(has_def)
            # SAB
            if "SAB" in config["attributes_features"]:
                source = np.stack(node.source)
                features_attributes.append(source)
            # Parents_Types
            if "Parents_Types" in config["attributes_features"]:
                if node.parents_type is not None and len(node.parents_type) > 0:
                    parents_type = " ".join(node.parents_type)
                else:
                    parents_type = ""
                parents_type_format = get_parents_type_format(
                    "TUI", [parents_type])
                features_attributes.append(np.stack(parents_type_format))

            attributes = np.concatenate(features_attributes, None)

            features.append(np.stack([attributes]))
        if "Labels" in config["attributes_features"]:
            features.append(np.stack([node.labels]))
        return features

    def save_prediction_to_ttl(self, source, config):
        print("Saving prediction to ttl")
        _code, onto = split_code_id(self.nodes[0].next[0].code_id)
        # Save the prediction to a ttl file
        f = open("artefact/predictions_" + str(source) + ".ttl", "w")
        namespace_ontology = "@prefix onto: <" + str(onto) + "> .\n"
        namespace_bpm = "@prefix bpm: <http://bioportal.bioontology.org/ontologies/umls/> .\n"
        namespace_sty = "@prefix sty: <http://purl.bioontology.org/ontology/STY/> .\n"
        f.write(namespace_ontology)
        f.write(namespace_bpm)
        f.write(namespace_sty)
        f.write("\n")

        def recusively_write_prediction(node, f):
            if node.prediction is not None:
                code, _onto = split_code_id(node.code_id)
                f.write("onto:" + str(code) +
                        " bpm:hasSTY sty:" + node.prediction + " .\n")
            for child in node.next:
                recusively_write_prediction(child, f)

        recusively_write_prediction(self.nodes[0], f)
        f.close()
        
    def recursively_get_node(self, node, code_id):
        if node.code_id == code_id:
            return node
        for child in node.next:
            result = self.recursively_get_node(child, code_id)
            if result is not None:
                return result
        return None
        
    def update_node_if_already_set(self, code_id, new_parents_code_id):
        node = self.recursively_get_node(self.nodes[0], code_id)
        if node is not None:
            node.parents_code_id.append(new_parents_code_id)
            return True
        return False
