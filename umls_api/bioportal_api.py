"""Python Simple API for the BioPortal API."""

import requests

from umls_api.metathesaurus_queries import get_umls_source_abreviation_nearby

def get_source(str):
    str = str[::-1]
    str = str.split('/')[1]
    return str[::-1]


class BioPortalAPI:
    """Class for the BioPortal API."""

    def __init__(self, api_key):
        """Initialize the BioPortalAPI class."""
        self.api_key = api_key
        self.base_url = "https://data.bioontology.org/ontologies/"
        self.source = None

    def request_url(self, url):
        try:
            if '?' in url:
                url += "&apikey=" + self.api_key
            else:
                url += "?apikey=" + self.api_key
            reponse = requests.get(url)
            return reponse.json()
        except Exception as e:
            print("request_url", str(e))
            return None

    def request_concept_by_source(self, source):
        """Get a concept by source."""
        try:
            url = self.base_url + str(source) + "/classes/roots"
            print("URL: ", url)
            return self.request_url(url)
        except Exception as e:
            print("request_concept_by_source", str(e))
            return None

    def get_concept_by_source(self, source):
        """Get a concept by source.

        Args:
            source (str): The source of the concept.

        Returns:
            dict: The concept.
        """
        try:
            results = self.request_concept_by_source(source)
            
            # Error handling
            if results is None:
                print("get_concept_by_source", "No results found.")
                return None

            return results
        except Exception as e:
            print("get_concept_by_source exception", str(e))
            return None

    def get_roots_of_tree(self, source):
        results = self.get_concept_by_source(source)
        if results is None:
            print("get_root_of_tree", "No results found.")
            return None
        return results

    def get_features_from_link(self, link, parents_code_id):
        results = self.request_url(link)
        if results is None:
            print("get_features_from_link", "No results found.")
            return None
        try:
            labels = " ".join([i for i in results['synonym']])
        except:
            labels = None
        try:
            definition = " ".join([i for i in results['definition']])
        except:
            definition = None
        json = {
            "pref_label": results['prefLabel'],
            "labels": results['prefLabel'] + ' ' + labels,
            "definition": definition,
            "has_definition": True if definition is not None and len(definition) > 0 else False,
            "source": self.source if self.source is not None else '',
            "parents_type": None,
            "semantic_type": results['semanticType'],
            "code_id": results['@id'],
            "parents_code_id": parents_code_id,
            "children": results['links']['children']
        }
        return json

    def get_children_links(self, children_collection_link):
        results = self.request_url(children_collection_link)
        if results is None:
            print("get_children_links", "No results found.")
            return None
        return [i['links']['self'] for i in results['collection']]
