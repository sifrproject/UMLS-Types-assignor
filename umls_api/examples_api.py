from umls_api.mysql_connection import db
from umls_api.metathesaurus_queries import MetathesaurusQueries
from umls_api.semantic_network_queries import SemanticNetworkQueries

if db.connect():
    
    # https://bioportal.bioontology.org/ontologies/STY/?p=classes&conceptid=root
    
    ######## STY Examples ########
    
    sty = SemanticNetworkQueries(db)
    
    # Every STY
    # res = sty.get_all_semantic_types()
    # for row in res:
    #     print(row)
    #     print("\n")

    # # STY by name
    # res = sty.get_semantic_type_by_name("Clinical Attribute")
    # for row in res:
    #     print(row)
    #     print("\n")
        
    # # Get last parent of STY by name
    # res = sty.get_last_parents_semantic_type_by_name("Clinical Attribute")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get all parents of STY by name
    # res = sty.get_all_parents_semantic_type_by_name("Clinical Attribute")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get children of STY by name
    # res = sty.get_childen_semantic_type_by_name("Pathologic Function")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get all children of STY by name
    # res = sty.get_all_children_semantic_type_by_name("Pathologic Function")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get siblings of STY by name
    # res = sty.get_siblings_semantic_type_by_name("Conceptual Entity")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get relationships of STY by name
    # res = sty.get_relationships_semantic_type_by_name("Disease or Syndrome")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get all relationships of STY by name
    # res = sty.get_all_relationships_semantic_type_by_name("Disease or Syndrome")
    # for row in res:
    #     print(row)
    #     print("\n")

    # # Get STY by TUI 
    # res = sty.get_semantic_type_by_tui("T191")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # ... Same examples with "by TUI"
    
    ##############################

    ######## Metathesaurus Examples ########
    
    meta = MetathesaurusQueries(db)
    
    # # Get all names of a concept given its CUI
    # res = meta.get_all_names_from_cui("C0018999")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get all STY of a concept given its name
    # res = meta.get_all_semantic_types_from_name("Air contaminant")
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get all concept
    # res = meta.get_all_mrconso()
    # for row in res:
    #     print(row)
    #     print("\n")
    
    # # Get all concept with a given semantic type
    # res = meta.get_all_mrcon_with_sty()
    # for row in res:
    #     print(row)
    #     print("\n")
    
    db.disconnect()