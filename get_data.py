# 1st Step : Get the data from the UMLS

import os
import sys
import timeit
from umls_api.column_type import ColumnType
from numpy import NaN
import pandas as pd
from umls_api.mysql_connection import db, UMLS_API_KEY
from umls_api.metathesaurus_queries import MetathesaurusQueries
from umls_api.semantic_network_queries import SemanticNetworkQueries
from umls_api.authentication_umls_api import Authentication


def print_description(concept_cui: str, concept_tui: str, concept_gui: str):
    """Print the description of a concept

    Args:
        concept_cui (str): CUI of the concept
        concept_tui (str): TUI of the concept
        concept_gui (str): GUI of the concept
    """
    print("\n")
    print("cui", concept_cui)
    print("tui", concept_tui)
    print("gui", concept_gui)
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
    data = pd.DataFrame(X_data, columns=['CUI', 'Corpus', 'Has_Definition', 'Prefered_Label',
                                         'Labels', 'SAB', 'Parents_Types'])
    data = pd.concat(
        [data, pd.DataFrame(Y_data, columns=['TUI', 'GUI'])], axis=1)
    # Save X and Y in the same csv file
    if db_index is not None and db_index > 0:
        data.to_csv('artefact/data.csv', mode="a", index=False, header=False)
    else:
        data.to_csv('artefact/data.csv', index=False)


def generate_source_data(limit: int, config: dict, verbose=False):
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

    gui_indexes = {}
    all_gui = sty.get_all_gui()
    for item in all_gui:
        gui_indexes[item[1]] = item[0]

    res = meta.get_all_unique_terms(nb_data=limit, offset=db_index)
    count = 0
    X_data = []
    Y_data = []
    try:
        for (cui, tui) in res:
            gui = gui_indexes[tui]
            if verbose:
                print_description(cui, tui, gui)

            if verbose:
                print("Getting labels...")
            try:
                prefered_label = meta.get_prefered_label_from_cui(cui)
                labels = " ".join(
                    list(map(lambda x: x[0], meta.get_all_labels_from_cui(cui))))
            except Exception as e:
                labels = ""
                print(str(e))
            if verbose:
                print("Done.")

            if verbose:
                print("Getting definition...")
            try:
                definitions = " ".join(
                    list(map(lambda x: x[0], meta.get_all_definitions_from_cui(cui))))
                has_definition = True
                if definitions == "":
                    has_definition = False

                # Corpus is the concatenation of prefered_labels and definitions
                corpus = definitions
            except Exception as e:
                definitions = ""
                has_definition = False
                print(str(e))
            if verbose:
                print("Done.")

            if verbose:
                print("Getting sources...")
            try:
                sources = meta.get_sources_from_cui(cui)
                sab = ""
                for i in sources:
                    if i[0]:
                        sab += i[0]
                    if i != sources[-1]:
                        sab += "/"
            except Exception as e:
                sab = ""
                print(str(e))
            if verbose:
                print("Done.")

            if verbose:
                print("Getting parents informations...")
            try:
                type = ColumnType.TUI if config["y_classificaton_column"] == "TUI" \
                    else ColumnType.GUI
                parents_types = meta.get_all_type_of_parent_from_cui(
                    cui, type, gui_indexes)
                parents_types = " ".join(parents_types)
            except Exception as e:
                parents_types = ""
                print(str(e))
            if verbose:
                print("Done.")

            row = [
                cui,
                corpus,
                1 if has_definition else 0,
                prefered_label,
                labels,
                sab,
                parents_types
            ]
            X_data.append(row)
            Y_data.append([tui, gui])
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
