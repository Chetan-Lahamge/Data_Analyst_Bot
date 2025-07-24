import pyodbc
import os
from openai import AsyncOpenAI
from openai import OpenAI
import config
import asyncio
from semantic_kernel.functions import kernel_function

class DataCatalogueAgent:
    def __init__(self, connection):
        self.connection = connection

    @kernel_function
    async def get_table_summaries(self, output_dir="LLM_summaries"):
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
                full_table_name = f"{table_name}"

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
                    summary = await self.generate_llm_summary(prompt)

                    # Save the summary to a file
                    summary_file_path = os.path.join(output_dir, f"{table_name}_summary.txt")
                    with open(summary_file_path, 'w') as file:
                        file.write(summary)

                    summaries[table_name] = summary
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

    @kernel_function
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

    @kernel_function
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

    @kernel_function
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

    @kernel_function
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

    @kernel_function
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

    @kernel_function
    def generate_llm_prompt(self, table_name, columns, relationships, top_rows):
        prompt=f"""
            You are tasked with analyzing the table '{table_name}' from the database. The table contains the following columns: {', '.join(columns)}.
            Here are the first few rows of data from the table to help you understand its structure and business context:
            
            Column Data: {top_rows}

            The table also has the following relationships (such as Primary Keys and Foreign Keys) that define its role in the database structure:
            
            Relationships: {relationships}

            Your task is to review this data and generate detailed descriptions for each column in a natural language format. 
            For each column, provide a description that explains what the column represents, its data type, and its role within the table.
            The description should reflect the meaning of the column as inferred from the data and its business context. 
            Focus on what the column represents and how it contributes to the overall function of the table.

            Use the following format for each column description:
            
            Column Name: <Descriptive sentence about the column>.

            Once the descriptions for each column are complete, generate a 3-4 sentence summary of the table's overall purpose. 
            This summary should explain how the table is used in a business context, describe how the columns work together, and highlight the key relationships between columns (like PK/FK relationships).

            Use the following format for the table description:
            
            Table Description: <3-4 sentence description of the table's purpose, columns, and business context>.

            Additionally, based on the column descriptions and table summary, provide 3 unique tag words that summarize this table's content and significance. 
            These tags should capture the essence of the table's data and business use case.

            Table Tags: Tag1, Tag2, Tag3
        """
        return prompt

    
    @kernel_function
    async def generate_llm_summary(self, prompt):
        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary."


def get_db_connection():
    """Establish database connection."""
    try:
        print("Attempting to connect to SQL Server...")
        # connection = pyodbc.connect(
        #     'DRIVER={ODBC Driver 17 for SQL Server};'
        #     'SERVER=localhost,1433;'
        #     'DATABASE=AdventureWorks2019;'
        #     'UID=sa;'
        #     'PWD=Ch3ckmat3;'
        #     'TrustServerCertificate=yes;'
        #     'Encrypt=no;'
        # )
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost,1433;'
            'DATABASE=adventureworks;'
            'UID=sa;'
            'PWD=Ch3ckm@t3;'
            'TrustServerCertificate=yes;'
            'Encrypt=no;'
        )
        print("Successfully connected to SQL Server")
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        # Display available drivers for debugging
        try:
            print("Available ODBC drivers:")
            for driver in pyodbc.drivers():
                print(f"  - {driver}")
        except Exception as driver_err:
            print(f"Error listing drivers: {driver_err}")
        return None
