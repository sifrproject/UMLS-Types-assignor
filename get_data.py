from itertools import count
import os
import sys
import timeit
from mysqlx import Auth
from numpy import NaN
import pandas as pd
from umls_api.mysql_connection import db, UMLS_API_KEY
from umls_api.metathesaurus_queries import MetathesaurusQueries
from umls_api.semantic_network_queries import SemanticNetworkQueries
from umls_api.authentication_umls_api import Authentication


def print_description(concept_label: str, concept_sab: str, concept_cui: str,
                      concept_sty_label: str, concept_tui: str):
    """Print the description of a concept

    Args:
        concept_label (str): _description_
        concept_sab (str): _description_
        concept_cui (str): _description_
        concept_sty_label (str): _description_
        concept_tui (str): _description_
    """
    print("\n")
    print("label", concept_label)
    print("sab", concept_sab)
    print("cui", concept_cui)
    print("sty_label", concept_sty_label)
    print("tui", concept_tui)
    print("\n")


def save_index_db(count: int, db_index_path="db_index"):
    """Save the index of the database in a file

    Args:
        count (int): Count of the database
        db_index_path (str, optional): Path of the file. Defaults to "db_index".
    """
    text_file = open(db_index_path, "w+")
    text_file.write(str(count))
    text_file.close()


def save_to_csv(X_data: list, Y_data: list, db_index: int):
    """Save the data to a csv file

    Args:
        X_data (list): X data
        Y_data (list): Y data
        db_index (int): Index of the database
    """
    data = pd.DataFrame(X_data, columns=['Label', 'Source', 'CUI', 'Nb_Parents', 'Nb_Children',
                                    'Nb_Parents_Children_Known', 'Definition', 'Has_Definition'])
    data = pd.concat([data, pd.DataFrame(Y_data, columns=['TUI', 'GUI'])], axis=1)
    # Save X and Y in the same csv file
    if db_index > 0:
        data.to_csv('data.csv', mode="a", index=False, header=False)
    else:
        data.to_csv('data.csv', index=False)


def main():
    """Main function"""
    db_index = None
    db_index_path = "db_index"

    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        # Reset db_index file
        save_index_db(0, db_index_path)
        # remove data.csv file
        if os.path.exists('data.csv'):
            os.remove('data.csv')
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage:")
        print("\tpython3 get_data.py [--help] [--index]")
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "--index":
        if os.path.isfile(db_index_path):
            text_file = open(db_index_path, "r")
            db_index = int(text_file.read())
            text_file.close()
            print("Index Database: ", db_index)
        else:
            print("No db_index file found")
            sys.exit(84)

    start = timeit.default_timer()

    meta = MetathesaurusQueries(db)
    sty = SemanticNetworkQueries(db)
    auth = Authentication(UMLS_API_KEY)

    gui_indexes = {}
    all_gui = sty.get_all_gui()
    for index, item in enumerate(all_gui):
        gui_indexes[item[1]] = item[0]

    res = meta.get_all_mrcon_with_sty(nb_data=100, offset=db_index)
    count = 0
    X_data = []
    Y_data = []
    try:
        for (label, sab, cui, lui, sty_label, tui) in res:
            print_description(label, sab, cui, sty_label, tui)

            source = sab
            aui = meta.get_aui_from_cui_and_source_and_lui(cui, source, lui)
            nb_parents = -1
            nb_children = -1
            if aui:
                nb_parents_children_known = 1
                print("Getting number of ancestors...")
                nb_parents = meta.get_nb_all_parents_from_aui_and_umls_api(
                    auth, aui)
                print("Getting number of descendants...")
                nb_children = meta.get_nb_all_children_from_aui_and_umls_api(
                    auth, aui)
                if nb_children == -1 and nb_parents == -1:
                    nb_parents_children_known = 0
                else:
                    nb_parents = 0 if nb_parents == -1 else nb_parents
                    nb_children = 0 if nb_children == -1 else nb_children
            else:
                nb_parents_children_known = 0

            print("Getting definition...")
            definition = meta.get_definition_from_cui_and_source(cui, source)
            print("Done.")
            has_definition = True
            if not definition:
                has_definition = False

            row = [
                label,
                sab,
                cui,
                nb_parents if nb_parents != -1 else NaN,
                nb_children if nb_children != -1 else NaN,
                1 if nb_parents_children_known else 0,
                definition,
                1 if has_definition else 0,
            ]
            X_data.append(row)
            Y_data.append([tui, gui_indexes[tui]])
            count += 1
            print("\n")
    except KeyboardInterrupt:
        print("Quitting...")
        save_to_csv(X_data, Y_data, db_index)
        save_index_db(db_index + count)
        sys.exit(0)

    print("Number of rows:", count)

    save_to_csv(X_data, Y_data, db_index)
    save_index_db(db_index + count)

    stop = timeit.default_timer()

    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
