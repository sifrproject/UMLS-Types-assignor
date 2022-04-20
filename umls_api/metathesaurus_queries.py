import json
import requests
from typing import Any, List, Tuple
from umls_api.authentication_umls_api import Authentication
from umls_api.mysql_connection import DatabaseConnection
from umls_api.languages import Languages


class MetathesaurusQueries:
    """Metathesaurus API"""

    def __init__(self, database: DatabaseConnection):
        """Constructor

        Args:
            database (DatabaseConnection): The database connection
        """
        self.db = database
        self.service = "https://uts-ws.nlm.nih.gov"

    def get_all_names_from_cui(self, cui: str, language=Languages.ENG,
                               all=True) -> List[Any]:
        """Returns all names of a concept given its CUI

        Returns:
            List[Any]: All names of the concept
        """
        query = "SELECT SAB, STR, LAT FROM MRCONSO WHERE CUI = '{}' AND stt = 'PF' AND ts = 'P' \
            AND lat='{}'".format(cui, language.value)
        return self.db.execute_query(query, all)

    def get_all_semantic_types_from_name(self, name: str, all=True) -> List[Tuple[str, str]]:
        """Returns all STY of a concept given its name

        Returns:
            List[(str, str)]: All STY of the concept. Each STY is a tuple of (Label, STY)
        """
        query = "SELECT sty, tui FROM MRCON a, MRSTY b WHERE a.cui = b.cui AND str = '{}'".format(
            name)
        return self.db.execute_query(query, all)

    def get_all_mrconso(self, language=Languages.ENG, all=True) -> List[Any]:
        """Returns all concepts

        Returns:
            List[Any]: All concepts
        """
        query = "SELECT * FROM MRCONSO WHERE LAT = '{}' <> 0;".format(
            language.value)
        return self.db.execute_query(query, all)

    def _sortTuple(self, tup: List[Tuple[str, str, str, str, str]]):
        """Sorts a list of tuples alphabetically

        Args:
            tup (List[Tuple[str, str, str, str, str]]): The list of tuples to sort

        Returns:
            List[Tuple[str, str, str, str, str]]: The list of tuples sorted alphabetically
        """
        # Getting the length of list
        # of tuples
        n = len(tup)

        for i in range(n):
            for j in range(n-i-1):

                if tup[j][1] > tup[j + 1][1]:
                    tup[j], tup[j + 1] = tup[j + 1], tup[j]

        return tup

    def _removeDuplicated(self, tup: List[Tuple[str, str, str, str, str]]):
        """Removes duplicated tuples

        Args:
            tup (List[Tuple[str, str, str, str, str]]): The list of tuples to remove duplicated

        Returns:
            List[Tuple[str, str, str, str, str]]: The list of tuples without duplicated
        """
        # Check if 2 elements has the same source, CUI and TUI in a row
        for index, item in enumerate(tup):
            if index > 0 and item[1] == tup[index - 1][1] and item[2] == tup[index - 1][2] \
                    and item[4] == tup[index - 1][4]:
                tup.pop(index)
        return tup

    def get_all_mrcon_with_sty(self, nb_data=0, language=Languages.ENG, all=True,
                               offset=None) -> List[Tuple[str, str, str, str, str]]:
        """Returns all concepts with STY

        Returns:
            List[Tuple[str, str, str, str, str]]: All concepts with STY. Each concept is a tuple \
                of (Label, SAB, CUI, LUI, STYLabel, TUI)
        """
        query = "SELECT a.str, c.sab, a.cui, a.lui, b.sty, b.tui FROM MRCON a, MRSTY b, MRSO c \
            WHERE LAT = '{}' AND a.cui=b.cui AND a.ts = 'P' AND a.stt = 'PF' \
                AND a.lui=c.lui".format(language.value)
        if nb_data != 0:
            query += " LIMIT " + str(nb_data)
            if offset:
                query += " OFFSET " + str(offset)
        res = self.db.execute_query(query, all)
        if res is None:
            return None
        alphabetic_sorted_tuples = self._sortTuple(res)
        return self._removeDuplicated(alphabetic_sorted_tuples)

    def get_aui_from_cui_and_source_and_lui(self, cui: str, source: str,
                                            lui: str, language=Languages.ENG) -> str:
        """Returns the AUI of a concept given its CUI, source and LUI

        Returns:
            str: The AUI of the concept
        """
        query = "SELECT AUI FROM MRCONSO WHERE CUI='{}' AND TS='P' AND LUI='{}' AND SAB='{}' \
            AND LAT='{}' LIMIT 1".format(cui, lui, source, language.value)
        res = self.db.execute_query(query, False)
        if res:
            return res[0]
        return None

    def get_source_from_cui(self, cui: str, auth: Authentication) -> str:
        """Returns the source of a concept given its CUI

        Returns:
            str: The source of the concept
        """
        query = {'ticket': auth.getst()}
        url = self.service + "/rest/content/current/CUI/" + cui + "/definitions"
        r = requests.get(url, params=query)
        items = json.loads(r.text)
        if 'error' in items or not 'result' in items or len(items['result']) == 0:
            return None
        return items['result'][0]['rootSource']

    def get_code_from_cui_and_source(self, cui: str, source: str) -> str:
        """Returns the code of a concept given its CUI and source

        Returns:
            str: The code of the concept
        """
        query = "SELECT CODE FROM MRSAT WHERE CUI = '{}' AND SAB = '{}' LIMIT 1".format(
            cui, source)
        res = self.db.execute_query(query, False)
        if res is None or len(res) == 0 or res[0] is None:
            print("No code found")
            return None
        return res[0]

    def get_definition_from_cui_and_source(self, cui: str, source: str) -> str:
        """Returns the definition of a concept given its CUI and source

        Returns:
            str: The definition of the concept
        """
        query = "SELECT DEF FROM MRDEF WHERE CUI = '{}' AND SAB = '{}' LIMIT 1".format(
            cui, source)
        res = self.db.execute_query(query, False)
        if res is None or len(res) == 0 or res[0] is None:
            query = "SELECT DEF FROM MRDEF WHERE CUI = '{}' AND SAB = 'MSH' LIMIT 1".format(
                cui)
            res = self.db.execute_query(query, False)
            if res is None or len(res) == 0 or res[0] is None:
                print("No definition found")
                return None
            return res[0]
        return res[0]

    def get_nb_all_children_from_aui_and_umls_api(self, auth: Authentication, aui: str) -> int:
        """Returns the number of descendants of a concept given its AUI and UMLS API

        Returns:
            int: The number of descendants of the concept
        """
        query = {'ticket': auth.getst(), 'pageSize': 100}
        url = self.service + "/rest/content/current/AUI/" + \
            aui + "/descendants"
        r = requests.get(url, params=query)
        items = json.loads(r.text)
        if 'error' in items or not 'result' in items or items['result'] is None:
            print(items)
            return -1
        return len(items['result'])

    def get_nb_all_parents_from_aui_and_umls_api(self, auth: Authentication, aui: str) -> int:
        """Returns the number of descendants of a concept given its AUI and UMLS API

        Returns:
            int: The number of descendants of the concept
        """
        query = {'ticket': auth.getst(), 'pageSize': 100}
        url = self.service + "/rest/content/current/AUI/" + \
            aui + "/ancestors"
        r = requests.get(url, params=query)
        items = json.loads(r.text)
        if 'error' in items or not 'result' in items or items['result'] is None:
            print(items)
            return -1
        return len(items['result'])

    def get_nb_all_parents_from_cui_and_aui_and_source(self, cui: str, aui: str,
                                                       source: str) -> int:
        """Returns the number of ancestors of a concept given its CUI, AUI and source

        Returns:
            int: The number of ancestors of the concept
        """
        query = "SELECT PTR from MRHIER WHERE CUI='{}' AND SAB='{}' AND AUI='{}' LIMIT 1".format(
            cui, source, aui)
        res = self.db.execute_query(query, False)
        return len(res[0].split("."))
