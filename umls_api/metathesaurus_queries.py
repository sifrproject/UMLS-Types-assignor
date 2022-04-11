import json
from typing import Any, List, Tuple
from umls_api.authentication_umls_api import Authentication
from umls_api.mysql_connection import DatabaseConnection
from umls_api.languages import Languages
import requests

class MetathesaurusQueries:

    # Constructor
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.service = "https://uts-ws.nlm.nih.gov"
        
    def get_all_names_from_cui(self, cui: str, language=Languages.ENG,all=True) -> List[Any]:
        """Returns all names of a concept given its CUI
        
        Returns:
            List[Any]: All names of the concept
        """
        query = "SELECT SAB, STR, LAT FROM MRCONSO WHERE CUI = '{}' AND stt = 'PF' AND ts = 'P' AND lat='{}'".format(cui, language.value)
        return self.db.execute_query(query, all)

    def get_all_semantic_types_from_name(self, name: str, all=True) -> List[Tuple[str, str]]:
        """Returns all STY of a concept given its name
        
        Returns:
            List[(str, str)]: All STY of the concept. Each STY is a tuple of (Label, STY)
        """
        query = "SELECT sty, tui FROM MRCON a, MRSTY b WHERE a.cui = b.cui AND str = '{}'".format(name)
        return self.db.execute_query(query, all)
    
    def get_all_mrconso(self, language=Languages.ENG, all=True) -> List[Any]:
        """Returns all concepts
        
        Returns:
            List[Any]: All concepts
        """
        query = "SELECT * FROM MRCONSO WHERE LAT = '{}' <> 0;".format(language.value)
        return self.db.execute_query(query, all)

    def get_all_mrcon_with_sty(self, language=Languages.ENG, all=True) -> List[Any]:
        """Returns all concepts with STY
        
        Returns:
            List[(str, str, str, str)]: All concepts with STY. Each concept is a tuple of (Label, CUI, STYLabel, TUI)
        """
        query = "SELECT a.str, a.cui, b.sty, b.tui FROM MRCON a, MRSTY b WHERE LAT = '{}' AND a.cui=b.cui AND a.ts = 'P' AND a.stt = 'PF' LIMIT 5".format(language.value)
        return self.db.execute_query(query, all)
    
    def get_source_from_cui(self, cui: str, auth: Authentication) -> str:
        """Returns the source of a concept given its CUI
        
        Returns:
            str: The source of the concept
        """
        query = {'ticket': auth.getst()}
        print(query)
        url = self.service + "/rest/content/current/CUI/" + cui + "/definitions"
        r = requests.get(url, params=query)
        items = json.loads(r.text)
        if len(items['result']) == 0:
            return None
        else:
            return items['result'][0]['rootSource']
    
    def get_code_from_cui_and_source(self, cui: str, source: str) -> str:
        """Returns the code of a concept given its CUI and source
        
        Returns:
            str: The code of the concept
        """
        query = "SELECT CODE FROM MRSAT WHERE CUI = '{}' AND SAB = '{}' LIMIT 1".format(cui, source)
        res = self.db.execute_query(query, False)
        if len(res) == 0:
            return None
        else:
            return res[0]
    
    def get_nb_all_parents_from_cui_from_umls_api(self, auth: Authentication, source: str, code: str) -> int:
        """Returns the number of ancestors of a concept given its source and code
        
        Returns:
            int: The number of ancestors of the concept
        """
        query = {'ticket': auth.getst(), 'pageSize': 100}
        print(query)
        url = self.service + "/rest/content/current/source/" + source + "/" + code + "/ancestors"
        r = requests.get(url, params=query)
        print(r)
        items = json.loads(r.text)
        print(items)
        if not items['result']:
            return 0
        else:
            return len(items['result']) - 2
        
    def get_nb_all_children_from_cui_from_umls_api(self, auth: Authentication, source: str, code: str) -> int:
        """Returns the number of descendants of a concept given its source and code
        
        Returns:
            int: The number of descendants of the concept
        """
        query = {'ticket': auth.getst(), 'pageSize': 100}
        print(query)
        url = self.service + "/rest/content/current/source/" + source + "/" + code + "/descendants"
        r = requests.get(url, params=query)
        items = json.loads(r.text)
        print(items)
        if not items['result']:
            return 0
        else:
            return len(items['result']) - 2
