from mysqlx import Auth
from numpy import NaN
from umls_api.mysql_connection import db, UMLS_API_KEY
from umls_api.metathesaurus_queries import MetathesaurusQueries
from umls_api.semantic_network_queries import SemanticNetworkQueries
from umls_api.authentication_umls_api import Authentication
import pandas as pd

if not db.connect():
    exit(84)

meta = MetathesaurusQueries(db)
sty = SemanticNetworkQueries(db)
auth = Authentication(UMLS_API_KEY)

def print_description(label, sab, cui, sty_label, tui):
    print("\n")
    print("label", label)
    print("sab", sab)
    print("cui", cui)
    print("sty_label", sty_label)
    print("tui", tui)
    print("\n")

tui_indexes = {}
all_tui = sty.get_all_tui()
for index, item in enumerate(all_tui):
    tui_indexes[item[0]] = index

res = meta.get_all_mrcon_with_sty(nb_data=10)
count = 0
X = []
Y = []
for (label, sab, cui, sty_label, tui) in res:
    print_description(label, sab, cui, sty_label, tui)

    source = sab
    code = meta.get_code_from_cui_and_source(cui, source)
    has_nb_parents_and_children = True
    if code == None:
        # Set firstly -1 to the nb_all_parents and nb_all_children
        # Then, we'll set the value to the mean of the nb_all_parents and nb_all_children
        nb_parents = -1
        nb_children = -1
        has_nb_parents_and_children = False
    else:
        nb_parents = meta.get_nb_all_parents_from_cui_from_umls_api(auth, source, code)
        nb_children = meta.get_nb_all_children_from_cui_from_umls_api(auth, source, code)

    definition = meta.get_definition_from_cui_and_source(cui, source)
    has_definition = True
    if not definition:
        has_definition = False

    if not tui in tui_indexes:
        print("Error: tui not found")
        print("tui", tui)
        exit(84)

    row = [
        label,                                      # 0 Label -> is deleted in the end because it is not useful right now
        sab,                                        # 0 SAB -> is deleted in the end because it is not useful
        cui,                                        # 0 CUI -> is deleted in the end because it is not useful
        nb_parents if nb_parents != -1 else NaN,    # 1 Number of parents
        nb_parents if nb_children != -1 else NaN,   # 2 Number of children
        1 if has_nb_parents_and_children else 0,    # 3 Has nb_parents and nb_children
        definition,                                 # 4 Definition
        1 if has_definition else 0,                 # 5 Has definition
    ]
    X.append(row)
    Y.append([tui, tui_indexes[tui]])
    count += 1

print("Number of rows:", count)
data = pd.DataFrame(X, columns=['Label', 'Source', 'CUI', 'Nb_Parents', 'Nb_Children', 'Has_Nb_Parents_And_Children', 'Definition', 'Has_Definition'])
data = pd.concat([data, pd.DataFrame(Y, columns=['TUI', 'TUI_Index'])], axis=1)

################ Nb_Parents and Nb_Children ###############

# Prepare the nb_parents and nb_children to be used in the model
mean_all_parents = data['Nb_Parents'].mean()
mean_all_children = data['Nb_Children'].mean()
print("Mean of parents", mean_all_parents)
print("Mean of children", mean_all_children)
data.loc[data['Nb_Parents'].isnull(), 'Nb_Parents'] = mean_all_parents
data.loc[data['Nb_Children'].isnull(), 'Nb_Children'] = mean_all_children
# Normalize the data
data['Nb_Parents'] = (data['Nb_Parents'] - data['Nb_Parents'].min()) / (data['Nb_Parents'].max() - data['Nb_Parents'].min())
data['Nb_Children'] = (data['Nb_Children'] - data['Nb_Children'].min()) / (data['Nb_Children'].max() - data['Nb_Children'].min())

###########################################################


################ Definition ##############################

# TODO : Use definition of the concept to predict the TUI

##########################################################

# Save X and Y in the same csv file
data.to_csv('data.csv', index=False)
