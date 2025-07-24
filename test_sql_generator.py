import asyncio
import sys
from Agents.sql_query_generator_agent import SQLQueryGeneratorAgent

async def test_sql_generator():
    # Initialize the SQL query generator agent
    sql_agent = SQLQueryGeneratorAgent()
    
    # Get the query from command line args or use default
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What are the top 5 customers by total order amount, and what products did they purchase in the last year?"
    
    print(f"Testing query: {test_query}")
    
    # Generate SQL query
    sql_query = await sql_agent.generate_sql_query(test_query)
    
    # Print the results
    print("\nGenerated SQL Query:")
    print("-" * 50)
    print(sql_query)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_sql_generator()) 
