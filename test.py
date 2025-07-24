# 1. Testing the schema.table_name format for SQL queries
'''
import pyodbc

def get_db_connection():
    """Establish a connection to the SQL Server database."""
    try:
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost,1433;'  # Update with your server details
            'DATABASE=AdventureWorks;'  # Update with your database name
            'UID=sa;'  # Update with your username
            'PWD=Ch3ckm@t3;'  # Update with your password
        )
        print("Connection successful!")
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def test_table_existence(connection):
    """Test basic queries to check table existence and schema."""
    try:
        cursor = connection.cursor()

        # Test query for Product table
        print("Testing Product table query...")
        cursor.execute("SELECT TOP 5 * FROM Production.Product")
        rows = cursor.fetchall()
        if rows:
            print("Product table exists and returned data.")

        # Test query for SalesOrderDetail table
        print("Testing SalesOrderDetail table query...")
        cursor.execute("SELECT TOP 5 * FROM Sales.SalesOrderDetail")
        rows = cursor.fetchall()
        if rows:
            print("SalesOrderDetail table exists and returned data.")

    except pyodbc.Error as e:
        print(f"Database error: {e}")

def main():
    connection = get_db_connection()
    if connection:
        test_table_existence(connection)
        connection.close()

if __name__ == "__main__":
    main()
'''



'''---------------------------------------------------------------------------------------------------'''

#2.Saving Table and correspondin column details
'''
import os
import pyodbc

def test_db_connection():
    try:
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost,1433;'
            'DATABASE=AdventureWorks;'
            'UID=sa;'
            'PWD=Ch3ckm@t3;'
        )
        cursor = connection.cursor()
        
        # Create the schema_details folder if it doesn't exist
        if not os.path.exists('schema_details'):
            os.makedirs('schema_details')

        # Query to get all table names and their schemas
        cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
        tables = cursor.fetchall()

        # Open a text file to write the detailed schema information
        schema_file_path = os.path.join('schema_details', 'adventureworks_schema.txt')
        with open(schema_file_path, 'w') as file:
            file.write("AdventureWorks Database Schema\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Total tables: {len(tables)}\n\n")

            # Open another text file to save table names in the desired format
            db_file_path = os.path.join('schema_details', 'db_names.txt')
            with open(db_file_path, 'w') as db_file:
                for table in tables:
                    schema_name = table.TABLE_SCHEMA
                    table_name = table.TABLE_NAME
                    formatted_name = f"dbo.{table_name}"  # e.g., dbo.Product
                    db_file.write(f"{table_name} : {formatted_name}\n")  # Write to db_names.txt

                    file.write(f"Table: {table_name}\n")

                    # Query to get the schema of each table
                    cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{table_name}'")
                    columns = cursor.fetchall()

                    file.write("Columns:\n")
                    for column in columns:
                        file.write(f"  - {column.COLUMN_NAME} ({column.DATA_TYPE})\n")

                    # Query for primary keys
                    cursor.execute(f"""
                        SELECT COLUMN_NAME 
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                        WHERE TABLE_NAME = '{table_name}' AND CONSTRAINT_NAME LIKE 'PK_%'
                    """)
                    primary_keys = cursor.fetchall()

                    if primary_keys:
                        file.write("Primary Keys:\n")
                        for pk in primary_keys:
                            file.write(f"  - {pk.COLUMN_NAME}\n")
                    else:
                        file.write("Primary Keys: None\n")

                    # Query for foreign keys
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, CONSTRAINT_NAME 
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                        WHERE TABLE_NAME = '{table_name}' AND CONSTRAINT_NAME LIKE 'FK_%'
                    """)
                    foreign_keys = cursor.fetchall()

                    if foreign_keys:
                        file.write("Foreign Keys:\n")
                        for fk in foreign_keys:
                            file.write(f"  - {fk.COLUMN_NAME} (Constraint: {fk.CONSTRAINT_NAME})\n")
                    else:
                        file.write("Foreign Keys: None\n")

                    file.write("\n" + "-" * 50 + "\n\n")

        connection.close()
        print("Database schema extracted successfully and saved to 'schema_details/adventureworks_schema.txt' and 'schema_details/db_names.txt'.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_db_connection()
'''





#3. Checking the table names
'''
import pyodbc

def test_db_connection():
    try:
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost,1433;'
            'DATABASE=AdventureWorks;'
            'UID=sa;'
            'PWD=Ch3ckm@t3;'
        )
        cursor = connection.cursor()
        
        # Query to get all table names
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
        tables = cursor.fetchall()
        
        # Print table names and count
        print(f"Total tables: {len(tables)}")
        for table in tables:
            print(f"Table: {table.TABLE_NAME}")

        connection.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_db_connection()
'''

'''---------------------------------------------------------------------------------------------------'''  

# 4. testing AzureOpenAI access
'''
import openai
from openai import AzureOpenAI
# Set your API key and endpoint
api_key = "f904028aa99c4118922ac5ac65f2acf3"  # Replace with your Azure OpenAI API key
endpoint = "https://adwendpoint.openai.azure.com/"  # Replace with your Azure OpenAI endpoint
deployment_name = "adwendpoint"  # Name of your model deployment (e.g., gpt-4, gpt-3.5-turbo)


client = AzureOpenAI(
    api_version="2023-07-01-preview",api_key=api_key,
    azure_endpoint= endpoint,
)
# Make a request to the Azure OpenAI API using chat completions
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "3 steps for a happy life",
        },
    ],
)

# Extract and print a legible response from the API
response_message = completion.choices[0].message.content
print(f"AI Response: {response_message}")
''' 


# 5. Data_catalogue agent with Azure openAi
'''
import pyodbc
import os
from openai import AzureOpenAI

class DataCatalogueAgent:
    def __init__(self, connection, azure_client):
        self.connection = connection
        self.azure_client = azure_client  # Add AzureOpenAI client for LLM integration

    def get_table_summaries(self, output_dir="LLM_summaries"):
        try:
            cursor = self.connection.cursor()
            
            # Retrieve all table names and schema names
            cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
            tables = cursor.fetchall()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            summaries = {}
            total_tables = len(tables)
            processed_tables = 0
            failed_tables = 0

            # For each table, retrieve schema details, relationships, and top 20 rows
            for table in tables:
                schema_name = table.TABLE_SCHEMA
                table_name = table.TABLE_NAME
                full_table_name = f"{schema_name}.{table_name}"

                print(f"Processing table: {full_table_name}...")

                try:
                    # Get column details and skip unsupported types
                    column_details = self.get_column_details(table_name, schema_name)

                    # Get foreign key/primary key relationships
                    relationship_summary = self.get_table_relationship_output(table_name, schema_name)

                    # Get top 20 rows for supported columns
                    top_rows = self.get_top_rows(table_name, schema_name, column_details['supported_columns'])

                    # Combine all information into a prompt for GPT-4
                    prompt = self.generate_llm_prompt(full_table_name, column_details['supported_columns'], relationship_summary, top_rows)

                    # Generate a human-readable summary using GPT-4
                    summary = self.generate_llm_summary(prompt)

                    # Save the summary to a file
                    summary_file_path = os.path.join(output_dir, f"{table_name}_summary.txt")
                    with open(summary_file_path, 'w') as file:
                        file.write(summary)

                    summaries[full_table_name] = summary
                    processed_tables += 1

                except Exception as e:
                    print(f"Error processing table {full_table_name}: {e}")
                    failed_tables += 1

            print(f"Processing complete: {processed_tables}/{total_tables} tables processed successfully.")
            if failed_tables > 0:
                print(f"{failed_tables} tables failed to process.")
            return summaries

        except Exception as e:
            print(f"Error retrieving table details: {e}")
            return None

    def get_column_details(self, table_name, schema_name):
        """Dynamically detect and exclude unsupported column types."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM [{schema_name}].[{table_name}] WHERE 1=0")  # Fetch just the schema
            columns = cursor.description

            supported_columns = []
            unsupported_columns = []
            for col in columns:
                column_name = col[0]

                # Attempt to fetch a few rows for this column to see if it's compatible
                try:
                    query = f"SELECT TOP 1 {column_name} FROM [{schema_name}].[{table_name}]"
                    cursor.execute(query)
                    cursor.fetchall()
                    supported_columns.append(column_name)
                except pyodbc.Error as e:
                    unsupported_columns.append(column_name)

            return {
                'supported_columns': supported_columns,
                'unsupported_columns': unsupported_columns
            }

        except Exception as e:
            print(f"Error retrieving column details for {table_name}: {e}")
            return None

    def get_top_rows(self, table_name, schema_name, supported_columns):
        """Fetch top 20 rows for supported columns."""
        try:
            cursor = self.connection.cursor()
            column_list = ', '.join(supported_columns)
            query = f"SELECT TOP 20 {column_list} FROM [{schema_name}].[{table_name}]"
            cursor.execute(query)
            rows = cursor.fetchall()

            formatted_rows = ""
            for row in rows:
                formatted_rows += ', '.join([str(value) for value in row]) + "\n"
            return formatted_rows

        except Exception as e:
            print(f"Error retrieving top rows for {table_name}: {e}")
            return "No data available."

    def get_table_relationship_output(self, table_name, schema_name):
        """Fetch table relationships (FK/PK) and handle nested relationships."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME FROM INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE WHERE TABLE_NAME='{table_name}' AND TABLE_SCHEMA='{schema_name}'")
            rows = cursor.fetchall()

            relationship_summary = ""
            for row in rows:
                if "FK" in row.CONSTRAINT_NAME:
                    dependent_table = self.get_foreign_key_relationship(table_name, row.COLUMN_NAME)
                    relationship_summary += f"Table {table_name} has a Foreign Key on column {row.COLUMN_NAME} linked with table {dependent_table}.\n"
                    relationship_summary += self.check_nested_table_relationship(dependent_table)
                elif "PK" in row.CONSTRAINT_NAME:
                    relationship_summary += f"Table {table_name} has a Primary Key on column {row.COLUMN_NAME}.\n"
            return relationship_summary if relationship_summary else "No relationships found."

        except Exception as e:
            print(f"Error retrieving relationships for {table_name}: {e}")
            return "Error retrieving relationships."

    def get_foreign_key_relationship(self, table_name, column_name):
        """Fetch details about the table linked via foreign key."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE WHERE COLUMN_NAME='{column_name}' AND TABLE_NAME!='{table_name}'")
            rows = cursor.fetchall()

            if rows:
                return rows[0].TABLE_NAME
            return "No linked table found."

        except Exception as e:
            print(f"Error retrieving FK relationship for {table_name}: {e}")
            return "Error retrieving FK relationship."

    def check_nested_table_relationship(self, table_name, visited=None):
        """Recursively check for nested table relationships (FK links), while preventing infinite recursion."""
        if visited is None:
            visited = set()

        # If the table has already been visited, stop recursion
        if table_name in visited:
            return ""

        # Mark the table as visited
        visited.add(table_name)

        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME FROM INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE WHERE TABLE_NAME='{table_name}'")
            rows = cursor.fetchall()

            nested_relationship_summary = ""
            for row in rows:
                if "FK" in row.CONSTRAINT_NAME:
                    nested_table = self.get_foreign_key_relationship(table_name, row.COLUMN_NAME)
                    nested_relationship_summary += f"Table {table_name} contains a Foreign Key on column {row.COLUMN_NAME}, further linked with {nested_table}.\n"
                    nested_relationship_summary += self.check_nested_table_relationship(nested_table, visited)

            return nested_relationship_summary if nested_relationship_summary else "No further nested relationships."

        except Exception as e:
            print(f"Error checking nested relationships for {table_name}: {e}")
            return "Error checking nested relationships."

    def generate_llm_prompt(self, table_name, columns, relationships, top_rows):
        """Generate the prompt to send to GPT-4 for summary generation."""
        prompt = f"""
        Generate a human-readable summary for the table '{table_name}'.
        The table has the following columns: {', '.join(columns)}.
        It has the following relationships: {relationships}.
        Here are some sample rows from the table: {top_rows}.
        """
        return prompt

    def generate_llm_summary(self, prompt):
        """Use GPT-4 to generate a detailed summary for the table."""
        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating summary using GPT-4: {e}")
            return "Error generating summary."


def get_db_connection():
    """Establish database connection."""
    try:
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost,1433;'
            'DATABASE=AdventureWorks;'
            'UID=sa;'
            'PWD=Ch3ckm@t3;'
        )
        return connection
    except Exception as e:
        print(f"Error: {e}")
        return None

'''

## 4.1 main() for the above .py file
''' 
from Agents.data_catalogue_agent import DataCatalogueAgent, get_db_connection
import config
from openai import AzureOpenAI
def get_azure_openai_client():
    """Set up AzureOpenAI client using config variables."""
    try:
        client = AzureOpenAI(
            api_version="2023-07-01-preview",
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        )
        return client
    except Exception as e:
        print(f"Error setting up Azure OpenAI client: {e}")
        return None

def main():
    # Establish database connection (already handled in DataCatalogueAgent)
    connection = get_db_connection()

    if connection:
        # Initialize the Azure OpenAI client
        azure_client = get_azure_openai_client()

        if azure_client:
            # Initialize the DataCatalogueAgent with the connection and Azure client
            data_catalogue_agent = DataCatalogueAgent(connection, azure_client)
            
            # Get table summaries
            summaries = data_catalogue_agent.get_table_summaries()

            if summaries:
                for table, summary in summaries.items():
                    print(f"{table}: {summary}")
            else:
                print("No summaries generated.")
        else:
            print("Failed to set up Azure OpenAI client.")
    else:
        print("Failed to connect to the database.")

if __name__ == "__main__":
    main()
'''


#5. Code to save the LLM_summaries

'''
import pyodbc
import os
import asyncio
from openai import AsyncOpenAI
import config
from semantic_kernel.functions import kernel_function
from Agents.data_catalogue_agent import get_db_connection,  DataCatalogueAgent
 
 
async def main():
   # Establish database connection
   connection = get_db_connection()
  
   if connection:
       # Create an instance of DataCatalogueAgent
       data_catalogue_agent = DataCatalogueAgent(connection)
 
       # Call the get_table_summaries method to generate summaries
       await data_catalogue_agent.get_table_summaries(output_dir="LLM_summaries")
 
       # Close the connection after use
       connection.close()
 
if __name__ == "__main__":
   asyncio.run(main())

'''

#6. Testign data viz agent

import asyncio
import openai
import pandas as pd
import matplotlib.pyplot as plt
from Agents.data_extractor_agent import DataExtractorAgent 
from Agents.data_viz_agent import DataVizAgent
from Agents.sql_query_generator_agent import SQLQueryGeneratorAgent
from Agents.data_catalogue_agent import get_db_connection

# Example usage:
async def main():
    # Establish the database connection
    connection = get_db_connection()

    # Initialize the agents
    sql_gen_agent = SQLQueryGeneratorAgent()  # Your SQLQueryGeneratorAgent
    data_extractor_agent = DataExtractorAgent(connection=connection)
    data_viz_agent = DataVizAgent(data_extractor_agent)

    # Get user input
    print("Welcome to the Data Analyst Wizard!")
    user_input = input("Please enter your query: ")

    # Generate SQL query using SQLQueryGeneratorAgent
    sql_query = await sql_gen_agent.generate_sql_query(user_input)
    print(f"Generated SQL Query: {sql_query}")

    # Execute the SQL query and get the DataFrame
    df = await data_viz_agent.execute_sql_query(sql_query)

    # Determine the appropriate plot type based on the query and data
    await data_viz_agent.determine_plot_type(df, sql_query)

if __name__ == "__main__":
    asyncio.run(main())
