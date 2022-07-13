"""Python Simple API for the BioPortal API."""

import requests

from umls_api.metathesaurus_queries import get_umls_source_abreviation_nearby

def get_code_id(str):
    """Unique concept code from BioPortal link @id concept"""
    return str.split('#')[-1][1:]

def get_source(str):
    str = str[::-1]
    str = str.split('/')[1]
    return str[::-1]


class BioPortalAPI:
    """Class for the BioPortal API."""

    def __init__(self, api_key):
        """Initialize the BioPortalAPI class."""
        self.api_key = api_key
        self.base_url = "https://data.bioontology.org/"
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

    def request_concept_by_name_and_source(self, name, source):
        """Get a concept by name and source."""
        try:
            url = self.base_url + "search?q=" + name
            if source is not None:
                url += "&ontologies=" + source
            print("URL: ", url)
            return self.request_url(url)
        except Exception as e:
            print("request_concept_by_name_and_source", str(e))
            return None

    def get_concept_by_name_and_source(self, name, source):
        """Get a concept by name and source.

        Args:
            name (str): The name of the concept.
            source (str): The source of the concept.

        Returns:
            dict: The concept.
        """
        try:
            results = self.request_concept_by_name_and_source(name, source)
            
            # Error handling
            if results is not None and "errors" in results and "Notice that acronyms are case sensitive" in results['errors'][0]:
                print("Error acronym case sensitive")
                results = self.request_concept_by_name_and_source(name, None)
            if results is not None and "collection" in results and len(results['collection']) == 0:
                results = self.request_concept_by_name_and_source(name, None)
            if results is None or not "collection" in results:
                print("get_concept_by_name_and_source", "No results found.")
                return None

            list_sources = []
            for i in range(len(results['collection'])):
                result = results['collection'][i]
                if "cui" not in result or len(result['cui']) == 0: # Only if res is in UMLS
                    continue
                else:
                    source = get_source(result['@id'])
                    list_sources.append({"source_name": source, "result_id": i})

            if len(list_sources) == 0: # Error handling
                print("No results found for", name, source)
                return None

            for i in range(len(list_sources)):
                print(str(i) + " => " + list_sources[i]["source_name"])
            source_id = input("Select source: ")

            try:
                source_id = int(source_id)
            except Exception as e:
                print("No source selected")
                return None

            return results["collection"][list_sources[source_id]["result_id"]]
        except Exception as e:
            print("get_concept_by_name_and_source exception", str(e))
            return None

    def get_root_of_tree(self, name, source):
        results = self.get_concept_by_name_and_source(name, source)
        if results is None:
            print("get_root_of_tree", "No results found.")
            return None
        bioportal_source_abreviation = get_source(results['@id'])
        umls_source_abreviation = get_umls_source_abreviation_nearby(bioportal_source_abreviation)
        self.source = umls_source_abreviation
        print("Source selected: " + str(bioportal_source_abreviation) + ' -> ' + str(self.source))
        return results['links']['self']

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
            "source": self.source if self.source is not None else None,
            "parents_type": None,
            "code_id": get_code_id(results['@id']),
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
