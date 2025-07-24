import pandas as pd
from Agents.data_extractor_agent import get_connection

def simple_query_debugger():
    # Get connection
    conn = get_connection()
    print("Connected to database successfully!")
    
    # Get query from user
    sql_query = input("Enter SQL query to execute: ")
    
    # Execute query
    try:
        print(f"\nExecuting: {sql_query}\n")
        df = pd.read_sql(sql_query, conn)
        print("\nResults:")
        print(df)
    except Exception as e:
        print(f"\nError executing query: {e}")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    simple_query_debugger() 
