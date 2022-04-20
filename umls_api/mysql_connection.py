import os
from dotenv import load_dotenv
from pathlib import Path

import pymysql
import pymysql.cursors

# Import the .env file
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

HOST = os.getenv('HOST')
USER = "root"
PASSWORD = os.getenv('PASSWORD')
DATABASE = os.getenv('DATABASE')
UMLS_API_KEY = os.getenv('UMLS_API_KEY')


class DatabaseConnection:
    # Constructor
    def __init__(self, host, user, password, database):
        self.host = host
        print("Host: " + self.host)
        self.user = user
        print("User: " + self.user)
        self.password = password
        print("Password: " + self.password)
        self.database = database
        print("Database: " + self.database)
        self.connector = None
        self.__connect()

    # Method to connect to database
    def __connect(self):
        """ Connect to MySQL database
        """
        connection = pymysql.connect(host=self.host,
                                     user=self.user,
                                     password=self.password,
                                     database=self.database,
                                     cursorclass=pymysql.cursors.SSCursor)
        self.connector = connection

    def is_query_modified_data(self, query: str) -> bool:
        """ Check if query is modifying data

        Args:
            query (str): query to check

        Returns:
            bool: True if query is modifying data, False otherwise
        """
        first_keyword = query.split(" ")[0]
        if (first_keyword == "SELECT" or first_keyword == "with"):
            return False
        else:
            return True

    # Method to execute query
    def execute_query(self, query: str, all=False):
        """ Execute query

        Args:
            query (str): query to execute
            all (bool, optional): True if returns all data False to return only one. Defaults to False.

        Returns:
            any: result of query
        """
        res = False
        try:
            with self.connector.cursor() as cursor:
                cursor.execute(query)
                if self.is_query_modified_data(query):
                    self.connector.commit()
                    res = True
                else:
                    if all:
                        rows = cursor.fetchall()
                        res = rows
                    else:
                        row = cursor.fetchone()
                        res = row
        except:
            res = False
            print("Error: unable to execute query", query)
        return res

    # Method to execute query
    def execute_query_with_limit(self, query: str, limit=10):
        """ Execute query with limit

        Args:
            query (str): query to execute
            limit (int, optional): limit of query. Defaults to 10.

        Returns:
            any: result of query
        """
        res = False
        try:
            with self.connector.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchmany(limit)
                res = rows
        except:
            res = False
            print("Error: unable to execute query", query)
        return res

    def __exit__(self):
        """ Close connection to database
        """
        self.connector.close()


db = DatabaseConnection(HOST, USER, PASSWORD, DATABASE)
