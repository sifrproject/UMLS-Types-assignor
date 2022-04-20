from typing import Any, List
from umls_api.mysql_connection import DatabaseConnection


class SemanticNetworkQueries:
    """Semantic Type API"""
    
    def __init__(self, database: DatabaseConnection):
        """Constructor

        Args:
            database (DatabaseConnection): Database connection
        """
        self.db = database

    def get_all_semantic_types(self) -> List[Any]:
        """Get all semantic types

        Returns:
            List[Any]: All semantic types
        """
        all = True
        query = "SELECT * FROM SRDEF WHERE rt = 'STY'"
        return self.db.execute_query(query, all)

    ######## BY NAME ########

    def get_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get semantic type by name

        Returns:
            List[Any]: Semantic type
        """
        query = "SELECT * FROM SRDEF WHERE sty_rl = '{}' AND rt = 'STY'".format(
            sty_rl)
        return self.db.execute_query(query, all)

    def get_last_parents_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get last parents semantic type by name

        Returns:
            List[Any]: Last parents semantic type
        """
        query = "SELECT b.* FROM SRSTR a, SRDEF b WHERE sty_rl1 = '{}' AND rl = 'isa' \
            AND b.rt = 'STY' AND a.sty_rl2 = b.sty_rl".format(sty_rl)
        return self.db.execute_query(query, all)

    def get_all_parents_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get all parents semantic type by name

        Returns:
            List[Any]: All parents semantic type
        """
        query = "SELECT b.* FROM SRSTRE2 a, SRDEF b WHERE sty1 = '{}' AND rl = 'isa' \
            AND b.rt = 'STY' AND a.sty2 = b.sty_rl ORDER BY stn_rtn".format(sty_rl)
        return self.db.execute_query(query, all)

    def get_childen_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get childen semantic type by name

        Returns:
            List[Any]: Childen semantic type
        """
        query = "SELECT b.* FROM SRSTR a, SRDEF b WHERE sty_rl2 = '{}' AND rl = 'isa' \
            AND b.rt = 'STY' AND a.sty_rl1 = b.sty_rl".format(sty_rl)
        return self.db.execute_query(query, all)

    def get_all_children_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get all children semantic type by name

        Returns:
            List[Any]: All children semantic type
        """
        query = "SELECT b.* FROM SRSTRE2 a, SRDEF b WHERE sty2 = '{}' AND rl = 'isa' \
            AND b.rt = 'STY' AND a.sty1 = b.sty_rl ORDER BY stn_rtn".format(sty_rl)
        return self.db.execute_query(query, all)

    # ! Does not work with "Entity" and "Event"
    # ? Create special condition for "Entity" and "Event" ?
    def get_siblings_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get siblings semantic type by name

        Returns:
            List[Any]: Siblings semantic type
        """
        query = "SELECT c.* FROM SRSTR a, SRSTR b, SRDEF c WHERE a.sty_rl1 = '{}' \
            AND a.rl = 'isa' AND b.rl = 'isa' AND a.sty_rl2 = b.sty_rl2 \
                AND a.sty_rl1 != b.sty_rl1 AND c.rt = 'STY' \
                    AND c.sty_rl = b.sty_rl1".format(sty_rl)
        return self.db.execute_query(query, all)

    def get_relationships_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get relationships semantic type by name

        Returns:
            List[Any]: Relationships semantic type
        """
        query = "SELECT a.rl, b.ui rl_ui, sty_rl2 sty, c.ui sty_ui FROM SRSTR a, SRDEF b, SRDEF c \
            WHERE sty_rl1 = '{}' AND ls = 'D' AND a.rl = b.sty_rl AND b.rt = 'RL' \
                AND a.sty_rl2 = c.sty_rl AND c.rt = 'STY'".format(sty_rl)
        return self.db.execute_query(query, all)

    def get_all_relationships_semantic_type_by_name(self, sty_rl: str, all=True) -> List[Any]:
        """Get all relationships semantic type by name

        Returns:
            List[Any]: All relationships semantic type
        """
        query = "SELECT a.rl, b.ui rl_ui, sty2 sty, c.ui sty_ui FROM SRSTRE2 a, SRDEF b, SRDEF c \
            WHERE sty1 = '{}' AND a.rl = b.sty_rl AND b.rt = 'RL' AND a.sty2 = c.sty_rl \
                AND c.rt = 'STY'".format(sty_rl)
        return self.db.execute_query(query, all)

    #########################

    ######## BY TUI #########

    def get_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get semantic type by TUI

        Returns:
            List[Any]: Semantic type
        """
        query = "SELECT * FROM SRDEF WHERE ui = '{}' AND rt = 'STY'".format(
            tui)
        return self.db.execute_query(query, all)

    def get_last_parents_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get last parents semantic type by TUI

        Returns:
            List[Any]: Last parents semantic type
        """
        query = "SELECT b.* FROM SRSTR a, SRDEF b, SRDEF c WHERE c.ui = '{}' \
            AND a.sty_rl1 = c.sty_rl AND rl = 'isa' AND b.rt = 'STY' \
                AND a.sty_rl2 = b.sty_rl AND c.rt = 'STY'".format(tui)
        return self.db.execute_query(query, all)

    def get_all_parents_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get all parents semantic type by TUI

        Returns:
            List[Any]: All parents semantic type
        """
        query = "SELECT b.* FROM SRSTRE1 a, SRDEF b WHERE ui1 = '{}' AND b.rt = 'STY' \
            AND a.ui3 = b.ui AND a.ui2 IN (SELECT ui FROM srdef WHERE rt= 'RL' \
                AND sty_rl = 'isa') ORDER BY stn_rtn".format(tui)
        return self.db.execute_query(query, all)

    def get_childen_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get childen semantic type by TUI

        Returns:
            List[Any]: Childen semantic type
        """
        query = "SELECT b.* FROM SRSTR a, SRDEF b, SRDEF c WHERE c.ui = '{}' \
            AND a.sty_rl2 = c.sty_rl AND rl = 'isa' AND b.rt = 'STY' \
                AND a.sty_rl1 = b.sty_rl AND c.rt = 'STY'".format(tui)
        return self.db.execute_query(query, all)

    def get_all_children_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get all children semantic type by TUI

        Returns:
            List[Any]: All children semantic type
        """
        query = "SELECT b.* FROM SRSTRE1 a, SRDEF b WHERE ui3 = '{}' AND b.rt = 'STY' \
            AND a.ui1 = b.ui AND a.ui2 IN (SELECT ui FROM srdef WHERE rt= 'RL' \
                AND sty_rl = 'isa') ORDER BY stn_rtn".format(tui)
        return self.db.execute_query(query, all)

    # * Same problem as with "get_siblings_semantic_type_by_name"
    def get_siblings_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get siblings semantic type by TUI

        Returns:
            List[Any]: Siblings semantic type
        """
        query = "SELECT c.* FROM SRSTR a, SRSTR b, SRDEF c, SRDEF d WHERE d.ui = '{}' \
            AND a.rl = 'isa' AND b.rl = 'isa' AND a.sty_rl2 = b.sty_rl2 \
                AND a.sty_rl1 != b.sty_rl1 AND c.rt = 'STY' AND b.sty_rl1 = c.sty_rl \
                    AND d.rt = 'STY' AND a.sty_rl1 = d.sty_rl".format(tui)
        return self.db.execute_query(query, all)

    def get_relationships_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get relationships semantic type by TUI

        Returns:
            List[Any]: Relationships semantic type
        """
        query = "SELECT a.rl, b.ui rl_ui, sty_rl2 sty, c.ui sty_ui \
            FROM SRSTR a, SRDEF b, SRDEF c, SRDEF d \
                WHERE d.ui = '{}' AND ls = 'D' AND a.rl = b.sty_rl AND b.rt = 'RL' \
                    AND a.sty_rl2 = c.sty_rl AND c.rt = 'STY' AND a.sty_rl1 = d.sty_rl \
                        AND d.rt = 'STY'".format(tui)
        return self.db.execute_query(query, all)

    def get_all_relationships_semantic_type_by_tui(self, tui: str, all=True) -> List[Any]:
        """Get all relationships semantic type by TUI

        Returns:
            List[Any]: All relationships semantic type
        """
        query = "SELECT a.rl, b.ui rl_ui, sty_rl2 sty, c.ui sty_ui FROM SRSTR a, SRDEF b, \
            SRDEF c, SRDEF d WHERE d.ui = '{}' AND ls = 'D' AND a.rl = b.sty_rl AND b.rt = 'RL' \
                AND a.sty_rl2 = c.sty_rl AND c.rt = 'STY' AND a.sty_rl1 = d.sty_rl \
                    AND d.rt = 'STY'".format(tui)
        return self.db.execute_query(query, all)

    #########################

    def get_all_tui(self, all=True) -> List[Any]:
        """Get all TUI

        Returns:
            List[Any]: All TUI
        """
        query = "SELECT UI FROM SRDEF WHERE RT='STY' ORDER BY UI"
        return self.db.execute_query(query, all)

    def get_all_gui(self, all=True) -> List[Any]:
        """Get all TUI

        Returns:
            List[Any]: All TUI
        """
        query = "SELECT SGA, TUI FROM SRGRP ORDER BY TUI"
        return self.db.execute_query(query, all)
