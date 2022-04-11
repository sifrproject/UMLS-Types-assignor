import os
from dotenv import load_dotenv
from pathlib import Path

import mysql.connector
from mysql.connector import Error

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
        self.connection = None
        self.cursor = None

    # Method to connect to database
    def connect(self):
        try:
            self.connection = mysql.connector.connect(host=self.host,
                                                      user=self.user,
                                                      password=self.password,
                                                      database=self.database)
            self.cursor = self.connection.cursor()
            print("Connected to MySQL database")
            return True
        except Error as e:
            print(e)
            print("Failed to connect to MySQL database")
            return False

    # Method to disconnect from database
    def disconnect(self):
        if self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("MySQL connection is closed")
            
    def is_query_modified_data(self, query):
        first_keyword = query.split(" ")[0]
        if (first_keyword == "SELECT"):
            return False
        else:
            return True
        

    # Method to execute query
    def execute_query(self, query, all=False):
        try:
            self.cursor.execute(query)
            if self.is_query_modified_data(query):
                self.connection.commit()
                return True
            else :
                if all:
                    rows = self.cursor.fetchall()
                    return rows
                else:
                    row = self.cursor.fetchone()
                    return row
        except Error as e:
            print(e)
            return None

    # Method to fetch all rows with limit
    def fetch_all_rows_with_limit(self, limit):
        try:
            self.cursor.execute("SELECT * FROM umls LIMIT %s", (limit,))
            rows = self.cursor.fetchall()
            return rows
        except Error as e:
            print(e)

db = DatabaseConnection(HOST, USER, PASSWORD, DATABASE)
