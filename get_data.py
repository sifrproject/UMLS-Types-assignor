# 1st Step : Get the data from the UMLS

import os
import sys
import timeit
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


def save_index_db(count: int, db_index_path="artefact/db_index"):
    """Save the index of the database in a file

    Args:
        count (int): Count of the database
        db_index_path (str, optional): Path of the file. Defaults to "db_index".
    """
    try:
        text_file = open(db_index_path, "w+", encoding='utf8')
        text_file.write(str(count))
        text_file.close()
    except Exception as e:
        print(str(e))


def save_to_csv(X_data: list, Y_data: list, db_index: int):
    """Save the data to a csv file

    Args:
        X_data (list): X data
        Y_data (list): Y data
        db_index (int): Index of the database
    """
    data = pd.DataFrame(X_data, columns=['Label', 'Source', 'CUI',
                                         'Nb_Parents', 'Nb_Children', 'Nb_Parents_Children_Known',
                                         'Definition', 'Has_Definition'])
    data = pd.concat(
        [data, pd.DataFrame(Y_data, columns=['TUI', 'GUI'])], axis=1)
    # Save X and Y in the same csv file
    if db_index is not None and db_index > 0:
        data.to_csv('artefact/data.csv', mode="a", index=False, header=False)
    else:
        data.to_csv('artefact/data.csv', index=False)


def generate_source_data(limit: int, verbose=False):
    """generate_source_data function"""
    db_index = None
    db_index_path = "artefact/db_index"

    if limit:
        if os.path.isfile(db_index_path):
            text_file = open(db_index_path, "r", encoding='utf8')
            db_index = int(text_file.read())
            text_file.close()
            print("Index Database: ", db_index)
        else:
            text_file = open(db_index_path, "w+", encoding='utf8')
            text_file.write('0')
            text_file.close()
            db_index = 0

    start = timeit.default_timer()

    meta = MetathesaurusQueries(db)
    sty = SemanticNetworkQueries(db)
    auth = Authentication(UMLS_API_KEY)

    gui_indexes = {}
    all_gui = sty.get_all_gui()
    for item in all_gui:
        gui_indexes[item[1]] = item[0]

    # res = meta.get_all_unique_terms() # ! delete me
    # deleteme = []
    # # Save res in a csv file
    # for i in res:
    #     deleteme.append([
    #         i[0],
    #         i[1],
    #         i[2],
    #         i[3],
    #         i[4],
    #         gui_indexes[i[4]],
    #     ])
    # data = pd.DataFrame(deleteme, columns=['CUI', 'AUI', 'SAB', 'STR', 'TUI', 'GUI'])
    # data.to_csv('artefact/analyse.csv', index=False)

    # exit(0)

    res = meta.get_all_mrcon_with_sty(nb_data=limit, offset=db_index)
    count = 0
    X_data = []
    Y_data = []
    try:
        for (label, sab, cui, lui, sty_label, tui) in res:
            if verbose:
                print_description(label, sab, cui, sty_label, tui)

            source = sab
            aui = meta.get_aui_from_cui_and_source_and_lui(cui, source, lui)
            nb_parents = -1
            nb_children = -1
            if aui:
                nb_parents_children_known = 1
                if verbose:
                    print("Getting number of ancestors...")
                nb_parents = meta.get_nb_all_parents_from_aui_and_umls_api(
                    auth, aui)
                if verbose:
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

            if verbose:
                print("Getting definition...")
            definition = meta.get_definition_from_cui_and_source(cui, source)
            if verbose:
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
            if verbose:
                print("\n")
    except KeyboardInterrupt:
        print("Quitting...")
        save_to_csv(X_data, Y_data, db_index)
        if limit:
            save_index_db(db_index + count)
        sys.exit(0)

    save_to_csv(X_data, Y_data, db_index)
    if limit:
        save_index_db(db_index + count)

    stop = timeit.default_timer()

    print(f'Generate source data in {stop - start} seconds')
    return (X_data, Y_data)
