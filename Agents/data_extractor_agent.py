import pyodbc
import pandas as pd
import re
from semantic_kernel.functions import kernel_function

class DataExtractorAgent:
    def __init__(self, connection):
        self.connection = connection
        self.table_schemas = self.get_table_schemas()

    @kernel_function
    def get_table_schemas(self):
        """Retrieve the schema for each table in the database."""
        schema_mapping = {}
        cursor = self.connection.cursor()

        # Fetch table names and their corresponding schemas
        cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
        tables = cursor.fetchall()

        # Map each table to its schema
        for table in tables:
            schema_mapping[table.TABLE_NAME] = table.TABLE_SCHEMA
        
        return schema_mapping

    @kernel_function
    def clean_query(self, sql_query):
        """Clean the SQL query of markdown and extra whitespace only, WITHOUT modifying table references."""
        # Remove markdown artifacts (```sql) and strip the query
        clean_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Clean excessive spaces
        clean_query = re.sub(r"\s+", " ", clean_query)
        
        print(f"Generated query: {sql_query}")
        print(f"Cleaned query for execution: {clean_query}")
        
        return clean_query

    @kernel_function
    def execute_query(self, sql_query):
        """Execute the SQL query on the database and return the results."""
        try:
            # Clean and validate the query before execution
            sql_query = self.clean_query(sql_query)

            # Print the cleaned query to ensure it's properly formatted
            print(f"Executing SQL Query: {sql_query}")

            cursor = self.connection.cursor()
            cursor.execute(sql_query)

            # Fetch all rows from the executed query
            rows = cursor.fetchall()

            # Fetch the column names from the cursor description
            columns = [column[0] for column in cursor.description]

            # Return data in tabular format using pandas
            return pd.DataFrame.from_records(rows, columns=columns)

        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None
