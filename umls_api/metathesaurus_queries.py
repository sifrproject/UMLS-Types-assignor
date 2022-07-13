from typing import Any, List, Tuple
import json
import urllib3
from umls_api.authentication_umls_api import Authentication
from umls_api.mysql_connection import DatabaseConnection
from umls_api.languages import Languages
from umls_api.column_type import ColumnType
from difflib import SequenceMatcher

f = open('umls_api/sources.json', 'r')
sources = json.load(f)
f.close()


def sort_tuple(tup: List[Tuple[str, str, str, str, str]]):
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


def remove_duplicated(tup: List[Tuple[str, str, str, str, str]]):
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


def get_tuple_of_languages_sources(language=Languages.ENG) -> Tuple[str]:
    """Gets the tuple of languages and sources

    Args:
        language (Languages, optional): The language to get the tuple of languages and sources. 
        Defaults to Languages.ENG.

    Returns:
        Tuple[str]: The tuple of sources abbreviation of the language
    """
    global sources
    return tuple([source['abbreviation'] for source in sources
                  if source['language'] == language.value])

def get_umls_source_abreviation_nearby(source_abreviation: str) -> str:
    """Gets the UMLS source abbreviation nearby

    Args:
        source_abreviation (str): The source abbreviation to get the UMLS source abbreviation nearby. 

    Returns:
        str: The UMLS source abbreviation nearby
    """
    global sources
    language=Languages.ENG
    scores = []
    for i in range(len(sources)):
        source = sources[i]
        if source['language'] == language.value:
            similarity = SequenceMatcher(None, source_abreviation, source['abbreviation']).ratio()
            scores.append({"score": similarity, "source": source['abbreviation']})
    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[0]['source']

class MetathesaurusQueries:
    """Metathesaurus API"""

    def __init__(self, database: DatabaseConnection):
        """Constructor

        Args:
            database (DatabaseConnection): The database connection
        """
        self.db = database
        self.service = "https://uts-ws.nlm.nih.gov"
        # Creating a PoolManager instance for sending requests.
        self.http = urllib3.PoolManager()

    def get_all_names_from_cui(self, cui: str, language=Languages.ENG,
                               all=True) -> List[Any]:
        """Returns all names of a concept given its CUI

        Returns:
            List[Any]: All names of the concept
        """
        query = f"SELECT SAB, STR, LAT FROM MRCONSO WHERE CUI = '{cui}' AND stt = 'PF' \
            AND ts = 'P' AND lat='{language.value}'"
        return self.db.execute_query(query, all)

    def get_all_semantic_types_from_name(self, name: str, all_rows=True) -> List[Tuple[str, str]]:
        """Returns all STY of a concept given its name

        Returns:
            List[(str, str)]: All STY of the concept. Each STY is a tuple of (Label, STY)
        """
        query = f"SELECT sty, tui FROM MRCON a, MRSTY b WHERE a.cui = b.cui AND str = '{name}'"
        return self.db.execute_query(query, all_rows)

    def get_all_mrconso(self, language=Languages.ENG, all_rows=True) -> List[Any]:
        """Returns all concepts

        Returns:
            List[Any]: All concepts
        """
        query = f"SELECT * FROM MRCONSO WHERE LAT = '{language.value}' <> 0"
        return self.db.execute_query(query, all_rows)

    def get_all_mrcon_with_sty(self, nb_data=0, language=Languages.ENG, all_rows=True,
                               offset=None) -> List[Tuple[str, str, str, str, str]]:
        """Returns all concepts with STY

        Returns:
            List[Tuple[str, str, str, str, str]]: All concepts with STY. Each concept is a tuple \
                of (Label, SAB, CUI, LUI, STYLabel, TUI)
        """
        query = f"SELECT a.str, c.sab, a.cui, a.lui, b.sty, b.tui FROM MRCON a, MRSTY b, MRSO c \
            WHERE LAT = '{language.value}' AND a.cui=b.cui AND a.ts = 'P' AND a.stt = 'PF' \
                AND a.lui=c.lui"
        if nb_data != 0 and nb_data is not None:
            query += " LIMIT " + str(nb_data)
            if offset:
                query += " OFFSET " + str(offset)
        res = self.db.execute_query(query, all_rows)
        if res is None:
            return None
        alphabetic_sorted_tuples = sort_tuple(res)
        return remove_duplicated(alphabetic_sorted_tuples)

    def get_aui_from_cui_and_source_and_lui(self, cui: str, source: str,
                                            lui: str, language=Languages.ENG) -> str:
        """Returns the AUI of a concept given its CUI, source and LUI

        Returns:
            str: The AUI of the concept
        """
        query = f"SELECT AUI FROM MRCONSO WHERE CUI='{cui}' AND TS='P' AND LUI='{lui}' \
            AND SAB='{source}' AND LAT='{language.value}' LIMIT 1"
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
        if query["ticket"] is None:
            return None
        url = self.service + "/rest/content/current/CUI/" + cui + "/definitions"
        resp = self.http.request("GET", url, fields=query)
        items = json.loads(resp.data)
        if 'error' in items or not 'result' in items or len(items['result']) == 0:
            return None
        return items['result'][0]['rootSource']

    def get_code_from_cui_and_source(self, cui: str, source: str) -> str:
        """Returns the code of a concept given its CUI and source

        Returns:
            str: The code of the concept
        """
        query = f"SELECT CODE FROM MRSAT WHERE CUI = '{cui}' AND SAB = '{source}' LIMIT 1"
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
        query = f"SELECT DEF FROM MRDEF WHERE CUI = '{cui}' AND SAB = '{source}' LIMIT 1"
        res = self.db.execute_query(query, False)
        if res is None or len(res) == 0 or res[0] is None:
            query = "SELECT DEF FROM MRDEF WHERE CUI = '{cui}' AND SAB = 'MSH' LIMIT 1"
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
        if query["ticket"] is None:
            return -1
        url = self.service + "/rest/content/current/AUI/" + \
            aui + "/descendants"
        resp = self.http.request("GET", url, fields=query)
        items = json.loads(resp.data)
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
        if query["ticket"] is None:
            return -1
        url = self.service + "/rest/content/current/AUI/" + \
            aui + "/ancestors"
        resp = self.http.request("GET", url, fields=query)
        items = json.loads(resp.data)
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
        query = f"SELECT PTR from MRHIER WHERE CUI='{cui}' AND SAB='{source}' \
            AND AUI='{aui}' LIMIT 1"
        res = self.db.execute_query(query, False)
        return len(res[0].split("."))

    def get_all_unique_terms(self, nb_data=0, offset=None) -> List[Tuple[str, str]]:
        """Returns all unique terms in the database

        Returns:
            list: A list of all unique terms [(CUI, TUI)]
        """
        query = "SELECT DISTINCT a.cui, b.tui from MRCONSO a, MRSTY b \
            WHERE a.cui=b.cui AND a.ISPREF='Y' AND a.LAT='ENG' AND a.STT='PF' AND a.TS='P'"
        if nb_data != 0 and nb_data is not None:
            query += " LIMIT " + str(nb_data)
            if offset:
                query += " OFFSET " + str(offset)
        res = self.db.execute_query(query, True)
        return res

    def get_all_definitions_from_cui(self, cui: str):
        """Returns all definitions of a concept given its CUI

        Args:
            cui (str): The CUI of the concept

        Returns:
            list: A list of all definitions [(DEF)]
        """
        list_eng_sources = get_tuple_of_languages_sources(Languages.ENG)
        query = f"SELECT def FROM MRDEF WHERE cui='{cui}' AND sab IN {list_eng_sources}"
        res = self.db.execute_query(query, True)
        return res

    def get_all_labels_from_cui(self, cui: str):
        """Returns all labels of a concept given its CUI

        Args:
            cui (str): The CUI of the concept

        Returns:
            list: A list of all labels [(STR)]
        """
        query = f"SELECT ANY_VALUE(str) FROM MRCONSO WHERE cui='{cui}' AND lat='ENG' \
            GROUP by lui  ORDER BY lui"
        res = self.db.execute_query(query, True)
        return res

    def get_prefered_label_from_cui(self, cui: str):
        """Returns the prefered label of a concept given its CUI

        Args:
            cui (str): The CUI of the concept

        Returns:
            str: The prefered label
        """
        query = f"SELECT ANY_VALUE(str) from MRCONSO WHERE cui='{cui}' AND lat='ENG' \
            AND ISPREF='Y' AND TS='P' GROUP BY lui ORDER BY lui"
        res = self.db.execute_query(query, True)
        try:
            return res[0][0]
        except IndexError:
            return ""

    def get_sources_from_cui(self, cui: str) -> List[Tuple[str, None]]:
        """Return all sources where the cui is listed

        Args:
            cui (str): The CUI of the concept
        Returns:
            List[Tuple[str, None]]: Sources
        """
        query = f"SELECT sab from MRCONSO WHERE cui='{cui}' AND LAT='ENG' GROUP BY sab"
        res = self.db.execute_query(query, True)
        try:
            return res
        except Exception as e:
            print(str(e))
            return [[]]

    def get_all_type_of_parent_from_cui(self, cui: str, type: ColumnType,
                                        gui_indexes: dict) -> List[str]:
        """Returns all parents type of a concept given its CUI

        Args:
            cui (str): The CUI of the concept

        Returns:
            list: A list of all TUI [(TUI)]
        """
        list_eng_sources = get_tuple_of_languages_sources(Languages.ENG)
        try:
            query = f"SELECT CUI1 FROM MRREL WHERE CUI2='{cui}' AND SAB IN {list_eng_sources} \
                AND REL='CHD' GROUP BY CUI1"
            cui_parents = self.db.execute_query(query, True)
            if cui_parents is None or len(cui_parents) == 0:
                return []
            for i in range(len(cui_parents)):
                cui_parents[i] = "'" + cui_parents[i][0] + "'"
            query = f"SELECT TUI FROM MRSTY WHERE CUI IN ({', '.join(cui_parents)})"
            res = self.db.execute_query(query, True)
            for i in range(len(res)):
                res[i] = res[i][0]
            if type == ColumnType.GUI:
                for i in range(len(res)):
                    res[i] = gui_indexes[res[i]]
            return res
        except Exception as e:
            print(str(e))
            return []
