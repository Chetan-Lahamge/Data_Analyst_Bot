import os
import re
import pickle
import faiss
import numpy as np
from openai import AsyncOpenAI, OpenAI
import config
from semantic_kernel.functions import kernel_function
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from Agents.data_catalogue_agent import get_db_connection

# Define paths for FAISS index and documents
# FAISS_INDEX_PATH = "faiss_index"
# DOCUMENTS_PATH = "faiss_documents.pkl"
FAISS_INDEX_PATH = "final_index"
DOCUMENTS_PATH = "final_output.pkl"

def get_few_shot_prompt(user_query: str) -> str:
    return f"""
Q: Why did revenue drop in Q1 2013?
SQL:
```sql
SELECT 
    DATEPART(QUARTER, soh.OrderDate) AS Quarter,
    DATEPART(YEAR, soh.OrderDate) AS Year,
    SUM(sod.LineTotal) AS Revenue
FROM Sales.SalesOrderHeader soh
JOIN Sales.SalesOrderDetail sod 
    ON soh.SalesOrderID = sod.SalesOrderID
WHERE YEAR(soh.OrderDate) IN (2012, 2013)
GROUP BY DATEPART(YEAR, soh.OrderDate), DATEPART(QUARTER, soh.OrderDate)
```sql
SELECT 
    DATEPART(QUARTER, soh.OrderDate) AS Quarter,
    DATEPART(YEAR, soh.OrderDate) AS Year,
    SUM(sod.LineTotal) AS Revenue
FROM Sales.SalesOrderHeader soh
JOIN Sales.SalesOrderDetail sod 
    ON soh.SalesOrderID = sod.SalesOrderID
WHERE YEAR(soh.OrderDate) IN (2012, 2013)
GROUP BY DATEPART(YEAR, soh.OrderDate), DATEPART(QUARTER, soh.OrderDate)
ORDER BY Year, Quarter;
```
Explanation:
This query calculates total revenue per quarter by joining order headers with order details to access LineTotal. It focuses on the years 2012 and 2013, allowing a clear comparison to identify any drop in Q1 2013.
Insights:
- Revenue in Q1 2013 was significantly lower than Q4 2012 and Q1 2012.
- The decline appears to be driven by reduced order volumes and lower average order values in January and February.
- March shows partial recovery but not enough to offset the earlier dip.

Recommendations:
- Introduce Q1 promotional campaigns to counteract post-holiday slumps.
- Analyze product mix in early 2013 to identify underperforming SKUs.
- Consider incentivizing early-year bulk orders through limited-time shipping discounts or loyalty points.

---

Q: Are there any surprises in my revenue metric in January of 2013?
SQL:
```sql
SELECT
    DATENAME(MONTH, soh.OrderDate) AS MonthName,
    DATEPART(MONTH, soh.OrderDate) AS MonthNum,
    DATEPART(YEAR, soh.OrderDate) AS Year,
    SUM(sod.LineTotal) AS TotalRevenue
FROM
    Sales.SalesOrderHeader soh
JOIN
    Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
WHERE
    soh.OrderDate BETWEEN '2012-07-01' AND '2013-06-30'
GROUP BY
    DATENAME(MONTH, soh.OrderDate),
    DATEPART(MONTH, soh.OrderDate),
    DATEPART(YEAR, soh.OrderDate)
ORDER BY
    Year, MonthNum;
```
Explanation:
This query retrieves monthly revenue from July 2012 to June 2013 to provide a broader context around January 2013. By comparing this month to both the 6 months prior and 5 months after, it becomes easier to determine if January's revenue behavior is an outlier.
Insights:
- Revenue in January 2013 was significantly lower than both December 2012 and February 2013.
- The decline may be due to post-holiday slowdown or operational adjustments at the start of the year.
- When compared to January 2012 (if available), the drop appears more pronounced, suggesting a possible deeper issue beyond seasonality.
- Marketing activity, product releases, or economic factors in early 2013 may have influenced performance.

Recommendations:
- Investigate if any post-holiday season return behavior or fulfillment issues occurred.
- Investigate campaign schedules, product stock levels, or any early Q1 operational changes. Consider replicating this monthly analysis across other years to validate if this dip is recurring or unique.
- Consider launching retention-focused email campaigns in early January next year to smooth out the drop.
- Explore if inventory gaps or delayed marketing campaigns contributed to the dip.

---

Q: Are there any notable trends in revenue for the year 2013?
SQL:
```sql
SELECT MONTH(OrderDate) AS Month, SUM(LineTotal) AS MonthlyRevenue
FROM Sales.SalesOrderHeader
WHERE YEAR(OrderDate) = 2013
GROUP BY MONTH(OrderDate)
ORDER BY Month;
```
Explanation:
This query captures monthly revenue trends for 2013.

Insights:
- Revenue was low in Q1, gradually increased in Q2, peaked in August, and dipped in October.
- July-August saw a 22% growth tied to back-to-school campaigns.
- October’s drop was linked to supply chain delays.

Recommendations:
- Maintain high inventory from June to August.
- Pre-order bulk stock for Q4 to avoid missed sales.

---

Q: What is the correlation between revenue and freight for year 2012?
SQL:
```sql
SELECT 
    MONTH(soh.OrderDate) AS Month, 
    SUM(sod.LineTotal) AS Revenue,
    SUM(soh.Freight) AS Freight
FROM Sales.SalesOrderHeader soh
JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
WHERE YEAR(soh.OrderDate) = 2012
GROUP BY MONTH(soh.OrderDate)
ORDER BY Month;
```
Explanation:
This query calculates total revenue and total freight charges for each month in 2012, helping assess if there's a correlation between how much customers spend and how much they pay for shipping.

Insights:
- A moderate positive correlation is observed: months with higher revenue also tend to have higher freight charges.
- For example, June and November show both high revenue and high freight, likely due to bulk orders or seasonal sales peaks.
- Months with unusually high freight but low revenue may indicate inefficient shipping strategies.

Recommendations:
- Consider offering free or discounted freight during high-revenue periods to further boost sales.
- Analyze order weight/volume trends to optimize shipping contracts or switch to better logistics providers.
- Use this freight-revenue correlation to segment customers by profitability after delivery costs.

"""


# few_shot_router.py
def matches_few_shot_case(user_query: str) -> bool:
    patterns = [
        # Revenue drop
        "why did revenue drop", "drop in revenue", "revenue declined", "revenue decrease", "fall in revenue",

        # Trends
        "trend in revenue", "notable trends", "revenue trend", "sales pattern", "sales trend", "any trends in",

        # Surprises or anomalies
        "surprises in revenue", "unexpected revenue", "anomaly in revenue", "spike in revenue", "dip in revenue",

        # Conversion rate
        "conversion rate normal", "is conversion rate normal", "conversion performance", "conversion drop",

        # Correlation
        "correlation between revenue and bounces", "correlation with bounces", "does bounce impact revenue",
        "relation between revenue and bounces", "bounce rate correlation"
    ]

    query = user_query.lower()
    return any(p in query for p in patterns)


class SQLQueryGeneratorAgent:
    def __init__(self, memory=None):
        self.memory = memory
        # Get database connection and load table schemas directly from database
        self.connection = get_db_connection()
        self.table_schemas = self.load_table_schemas_from_db()
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Check if FAISS index and documents exist
        self.has_embeddings = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH)
        if self.has_embeddings:
            print("Found embeddings for semantic search")
        else:
            print("No embeddings found - functionality will be limited")
    
    @kernel_function
    def load_summaries(self, summaries_dir):
        """Load table summaries from the LLM_summaries folder."""
        summaries = {}
        for filename in os.listdir(summaries_dir):
            if filename.endswith('_summary.txt'):
                table_name = filename.replace('_summary.txt', '')
                with open(os.path.join(summaries_dir, filename), 'r') as file:
                    summaries[table_name] = file.read()
        return summaries
    
    @kernel_function
    def load_table_schemas_from_db(self):
        """Load table schemas directly from the database."""
        table_schemas = {}
        if not self.connection:
            print("No database connection available")
            return table_schemas
            
        try:
            cursor = self.connection.cursor()
            
            # Query to get all tables and their schemas
            query = """
            SELECT 
                TABLE_NAME, 
                TABLE_SCHEMA 
            FROM 
                INFORMATION_SCHEMA.TABLES 
            WHERE 
                TABLE_TYPE = 'BASE TABLE'
            """
            
            cursor.execute(query)
            tables = cursor.fetchall()
            
            # Map each table to its schema
            for table in tables:
                table_name = table.TABLE_NAME
                schema_name = table.TABLE_SCHEMA
                table_schemas[table_name] = schema_name
                
            print(f"Loaded {len(table_schemas)} table schemas from database")
            return table_schemas
            
        except Exception as e:
            print(f"Error loading table schemas from database: {e}")
            return {}
    
    @kernel_function
    def load_relationships(self, schema_file="schema_details/adventureworks_schema.txt"):
        """Load schema relationships from a file."""
        relationships = {}
        try:
            with open(schema_file, 'r') as file:
                content = file.read()
                # Parse the content and extract relationships
                # For simplicity, just return the content for now
                return content
        except Exception as e:
            print(f"Error loading relationships: {e}")
            return ""
    
    def _extract_keywords(self, query):
        """Extract keywords from the query for search."""
        # Simple preprocessing to remove stop words and punctuation
        keywords = re.sub(r'[^\w\s]', '', query.lower())
        keywords = keywords.split()
        
        # Filter out common SQL-specific stop words
        sql_stop_words = set(['the', 'and', 'or', 'from', 'where', 'with', 'as', 'by', 'for', 'is', 'are', 'in', 'to', 'of', 'that', 'this', 'me', 'show', 'get', 'find', 'what', 'give', 'a', 'an'])
        keywords = [word for word in keywords if word not in sql_stop_words]
        
        print(f"Query keywords: {keywords}")
        return keywords
    
    def search_relevant_tables(self, query, n_results=5):
        """Search for relevant tables using embeddings or keywords."""
        if not self.has_embeddings:
            print("No embeddings available - using keyword search")
            # Fallback to keyword search
            return self._keyword_search(query, n_results)
        
        try:
            # Load faiss index and documents
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_PATH, 'rb') as f:
                documents = pickle.load(f)
            
            # Generate query embedding
            query_embedding = self.embed_model.get_text_embedding(query)
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # Search for similar documents
            scores, indices = faiss_index.search(query_embedding_np, n_results)
            
            # Extract and return the top matching documents
            results = {}
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    document = documents[idx]

                    # 🔧 FIXED: Check if it's a dict or a Document object
                    if isinstance(document, dict):
                        table_name = document.get("table_name", f"Unknown-{idx}")
                        summary = document.get("content", "No summary available")
                    else:
                        # Handle llama_index.Document
                        table_name = getattr(document, "metadata", {}).get("table_name", f"Unknown-{idx}")
                        summary = getattr(document, "text", "No summary available")

                    results[table_name] = summary
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            # Fallback to keyword search
            return self._keyword_search(query, n_results)
    
    def _keyword_search(self, query, n_results=5):
        """Simple keyword-based search for table relevance."""
        keywords = self._extract_keywords(query)
        if not keywords:
            return {}
        
        # Load all table summaries
        summaries = self.load_summaries("LLM_summaries")
        if not summaries:
            print("No summaries available for search")
            return {}
        
        # Score each summary based on keyword matches
        scores = {}
        for table, summary in summaries.items():
            score = 0
            for keyword in keywords:
                if keyword in summary.lower():
                    score += 1
            if score > 0:
                scores[table] = (score, summary)
        
        # Sort by score and take top n_results
        sorted_tables = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:n_results]
        
        # Return the results
        results = {}
        for table, (_, summary) in sorted_tables:
            results[table] = summary
        
        return results
    
    @kernel_function
    def construct_prompt(self, user_query, relevant_summaries=None):
        """Construct a prompt for the LLM to generate a SQL query."""
        table_references = []
        if relevant_summaries:
            for table, _ in relevant_summaries.items():
                if table in self.table_schemas:
                    # Use exactly the schema.table format from the database
                    schema = self.table_schemas[table]
                    table_references.append(f"{table}: {schema}.{table}")
        
        # Format the table reference examples
        table_examples = "\n".join(table_references)
        
        # Use correct SQL format examples
        # Use correct SQL format examples
        sql_format_examples = """
CORRECT: 
-- Simple filters
SELECT * FROM Purchasing.ProductVendor WHERE ProductID = 317

-- Basic JOIN
SELECT p.Name, od.OrderQty 
FROM Production.Product p 
JOIN Sales.SalesOrderDetail od ON p.ProductID = od.ProductID

-- ✅ Top-N per group using CROSS APPLY
SELECT
    st.Name AS TerritoryName,
    tp.Name AS ProductName,
    tp.ProductCategory,
    YEAR(tp.OrderDate) AS SalesYear,
    MONTH(tp.OrderDate) AS SalesMonth,
    SUM(tp.OrderQty) AS TotalQuantity,
    SUM(tp.LineTotal) AS TotalRevenue
FROM
    Sales.SalesTerritory st
CROSS APPLY (
    SELECT TOP 3
        p.Name,
        pc.Name AS ProductCategory,
        soh.OrderDate,
        sod.OrderQty,
        sod.LineTotal
    FROM Sales.SalesOrderHeader soh
    JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
    JOIN Production.Product p ON sod.ProductID = p.ProductID
    JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
    JOIN Production.ProductCategory pc ON psc.ProductCategoryID = pc.ProductCategoryID
    WHERE soh.TerritoryID = st.TerritoryID
      AND YEAR(soh.OrderDate) BETWEEN 2013 AND 2014
    ORDER BY sod.LineTotal DESC
) AS tp
GROUP BY
    st.Name, tp.Name, tp.ProductCategory, YEAR(tp.OrderDate), MONTH(tp.OrderDate)
ORDER BY
    st.Name, SalesYear, SalesMonth, TotalRevenue DESC
-- -- ✅ Correct: Join SalesPerson with Person.Person to get names
SELECT
    p.FirstName + ' ' + p.LastName AS SalesPersonName,
    SUM(sod.LineTotal) AS TotalRevenue
FROM
    Sales.SalesOrderHeader soh
JOIN
    Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
JOIN
    Sales.SalesPerson sp ON soh.SalesPersonID = sp.BusinessEntityID
JOIN
    Person.Person p ON sp.BusinessEntityID = p.BusinessEntityID
WHERE
    YEAR(soh.OrderDate) = 2014
GROUP BY
    p.FirstName, p.LastName
ORDER BY
    TotalRevenue DESC

-- ✅ Correct usage when needing Vendor Name
SELECT v.Name AS VendorName, AVG(pv.AverageLeadTime) AS AverageLeadTime
FROM Purchasing.ProductVendor pv
JOIN Purchasing.Vendor v ON pv.BusinessEntityID = v.BusinessEntityID
GROUP BY v.Name

INCORRECT (DO NOT USE THESE FORMATS):
-- ❌ Do not use dbo schema prefix
SELECT * FROM dbo.ProductVendor WHERE ProductID = 317

-- ❌ Do not omit schema name
SELECT * FROM ProductVendor WHERE ProductID = 317

-- ❌ Invalid schema format
SELECT * FROM dbo.Purchasing.ProductVendor WHERE ProductID = 317

-- ❌ Avoid HAVING clause with non-aggregated columns
SELECT ProductID, SUM(LineTotal) 
FROM Sales.SalesOrderDetail 
GROUP BY ProductID 
HAVING Sales.SalesTerritory.TerritoryID = 1  -- ❌ INVALID

-- ❌ Do NOT use pv.Name (invalid column)
SELECT pv.Name, AVG(pv.AverageLeadTime) FROM Purchasing.ProductVendor pv  -- ❌ Invalid

-- ❌ Incorrect: SalesPerson table has no FirstName or LastName
SELECT sp.FirstName FROM Sales.SalesPerson sp  -- ❌ Invalid

DO NOT USE CTEs (WITH clause) or multiple statements. Use simple, single SELECT statements.
"""

        
        prompt = f"""
You are an expert SQL query generator. Generate a SQL query to answer the user's question.

## TABLE FORMAT INSTRUCTIONS - CRITICAL REQUIREMENT
ALWAYS use the format 'SchemaName.TableName' for ALL table references. This is the ONLY valid format.
NEVER use 'dbo.TableName' format - this is INVALID and will cause errors.
NEVER use 'TableName' without schema - this is INVALID and will cause errors.
ALWAYS use the schema name provided in the table references below.

Examples:
{sql_format_examples}

## SQL STYLE REQUIREMENTS - CRITICAL
1. DO NOT use Common Table Expressions (CTEs) with the WITH clause
2. DO NOT use multiple SQL statements separated by semicolons
3. Use only a SINGLE, SIMPLE SELECT statement
4. For complex operations, use subqueries instead of CTEs
5. Keep queries compatible with basic SQL Server capabilities
#6. When generating queries or responding with data, always return and display the dimension name instead of the dimension ID. If a query involves a dimension stored as an ID, ensure #    it is joined with the appropriate table to retrieve the corresponding name. Never return or reference raw IDs unless explicitly requested. For example, when selecting columns## # #like    territory ID, product ID, location ID etc. use relevant tables such as Person.StateProvince, Production.Product and Production.Location and extract the name columns 

## Available Tables
These are the tables relevant to the user's query (format as SchemaName.TableName):
{table_examples}
## Few-Shot Examples (Learn from these)
These examples demonstrate how to handle observational or comparative questions like trends, revenue drops, normality, and correlation:

{get_few_shot_prompt(user_query)}

## Guidelines
1. Analyze the user's question to understand the required data
2. Use only the relevant tables listed above
3. MUST format ALL table references as 'SchemaName.TableName'
4. NEVER use 'dbo.TableName' format
5. Use proper table aliases where appropriate
6. Use column names exactly as specified in the table schemas
7. Structure the query for optimal performance
8. Avoid complex multi-part queries that use WITH clauses
#9. When generating queries or responding with data, always return and display the dimension name instead of the dimension ID. If a query involves a dimension stored as an ID, ensure #    it is joined with the appropriate table to retrieve the corresponding name. Never return or reference raw IDs unless explicitly requested. For example, when selecting columns #like    territory ID, product ID etc. use relevant tables such as Person.StateProvince and Production.Product and extract the name columns 


## User Question
{user_query}

## Your SQL Query
```sql
"""
        return prompt
    
    @kernel_function
    def fix_table_references(self, sql_query):
        """Make sure all table references use SchemaName.TableName format."""
        if not sql_query:
            return sql_query
        
        print(f"Original query before fixing references: {sql_query}")
        
        # We don't need to modify anything here anymore, 
        # as we're instructing the model to use SchemaName.TableName format properly
        # Just logging for debugging purposes
        
        print(f"Query after fixing references: {sql_query}")
        return sql_query
    
    @kernel_function
    async def generate_sql_query(self, user_query, conversation_history=None):
        """Generate a SQL query based on user query, using embeddings to find relevant tables."""
        # Check if this is a follow-up question
        is_followup = False
        previous_query = None
        previous_sql = None
        previous_tables = []
        
        if conversation_history and len(conversation_history) >= 2:
            # Enhanced heuristics to detect follow-up questions
            # 1. Explicit reference indicators
            followup_indicators = [
                'these', 'those', 'they', 'them', 'it', 'that', 'this',
                'show me more', 'drill down', 'what about', 'and how', 'compare', 
                'related', 'similar', 'same', 'like before', 'instead',
                'further', 'another', 'additional', 'more', 'less', 'fewer',
                'higher', 'lower', 'better', 'worse', 'previous', 'next'
            ]
            
            # 2. Check for questions that are too short to be standalone
            is_short_query = len(user_query.split()) <= 5
            
            # 3. Check for missing explicit subject in the query
            missing_subject = True
            subject_indicators = ['sales', 'revenue', 'product', 'customer', 'order', 'employee', 
                                  'inventory', 'stock', 'price', 'cost', 'profit', 'region', 'date']
            for indicator in subject_indicators:
                if indicator in user_query.lower():
                    missing_subject = False
                    break
            
            # Determine if it's a follow-up based on multiple factors
            if (any(indicator in user_query.lower() for indicator in followup_indicators) or 
                (is_short_query and missing_subject)):
                is_followup = True
                
                # Get relevant context from conversation history
                # Find the most recent user query
                for i in range(len(conversation_history)-1, -1, -1):
                    msg = conversation_history[i]
                    if msg.get('role') == 'user' and msg.get('content') != user_query:
                        previous_query = msg.get('content')
                        break
                
                # Find the most recent SQL query
                for i in range(len(conversation_history)-1, -1, -1):
                    msg = conversation_history[i]
                    if msg.get('role') == 'assistant' and 'Generated SQL Query' in msg.get('content', ''):
                        content = msg.get('content', '')
                        sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
                        if sql_match:
                            previous_sql = sql_match.group(1)
                            
                            # Extract tables used in the previous query for better context
                            table_pattern = r'FROM\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)|JOIN\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)'
                            table_matches = re.findall(table_pattern, previous_sql, re.IGNORECASE)
                            for match in table_matches:
                                for table in match:
                                    if table and '.' in table:
                                        schema_table = table.strip()
                                        table_name = schema_table.split('.')[1]
                                        if table_name not in previous_tables:
                                            previous_tables.append(table_name)
                        break
        
        # Search for relevant tables
        # For follow-up questions, include previously used tables in the search
        if is_followup and previous_tables:
            # Augment the user query with previous context for better table search
            search_query = f"{user_query} {' '.join(previous_tables)}"
            relevant_summaries = self.search_relevant_tables(search_query)
        else:
            relevant_summaries = self.search_relevant_tables(user_query)
        
        if is_followup and previous_query and previous_sql:
            # For follow-up questions, use the enhanced specialized prompt
            prompt = self.construct_followup_prompt(
                user_query, 
                previous_query, 
                previous_sql, 
                relevant_summaries,
                previous_tables
            )
        else:
            # For new questions, use the standard prompt
            prompt = self.construct_prompt(user_query, relevant_summaries)
        
        # Generate SQL query
        sql_query, confidence_score = await self.call_llm_to_generate_sql(prompt)
        
        # Apply table reference fix to ensure SchemaName.TableName format
        sql_query = self.fix_table_references(sql_query)
        
        return {
            "sql_query": sql_query,
            "relevant_tables": list(relevant_summaries.keys()) if relevant_summaries else [],
            "confidence_score": confidence_score
        }
    
    @kernel_function
    def construct_followup_prompt(self, user_query, previous_query, previous_sql, relevant_summaries=None, previous_tables=None):
        """Construct a prompt for follow-up questions with enhanced context awareness."""
        table_references = []
        if relevant_summaries:
            for table, summary in relevant_summaries.items():
                if table in self.table_schemas:
                    # Use correct schema.table format
                    schema = self.table_schemas[table]
                    table_references.append(f"{table}: {schema}.{table}")
                    # Include a brief table summary for better context
                    if summary:
                        # Extract first 2 sentences for brevity
                        brief_summary = '. '.join(summary.split('. ')[:2]) + '.'
                        table_references.append(f"  - Contains: {brief_summary}")
        
        # Format the table reference examples
        table_examples = "\n".join(table_references)
        
        # Analyze the previous SQL query to understand what it was doing
        previous_sql_analysis = """
This query retrieved:
"""
        # Extract SELECT fields
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_match = re.search(select_pattern, previous_sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            fields = select_match.group(1).strip()
            previous_sql_analysis += f"- Fields: {fields}\n"
        
        # Extract JOINs
        join_pattern = r'JOIN\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)'
        joins = re.findall(join_pattern, previous_sql, re.IGNORECASE)
        if joins:
            previous_sql_analysis += f"- Joined with: {', '.join(joins)}\n"
        
        # Extract WHERE conditions
        where_pattern = r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|HAVING|$)'
        where_match = re.search(where_pattern, previous_sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            conditions = where_match.group(1).strip()
            previous_sql_analysis += f"- Filtered by: {conditions}\n"
        
        # Extract GROUP BY fields
        group_pattern = r'GROUP BY\s+(.*?)(?:ORDER BY|HAVING|$)'
        group_match = re.search(group_pattern, previous_sql, re.IGNORECASE | re.DOTALL)
        if group_match:
            grouping = group_match.group(1).strip()
            previous_sql_analysis += f"- Grouped by: {grouping}\n"
        
        # Correct SQL format examples
        sql_format_examples = """
CORRECT: 
SELECT * FROM Purchasing.ProductVendor WHERE ProductID = 317
SELECT p.Name, od.OrderQty FROM Production.Product p JOIN Sales.SalesOrderDetail od ON p.ProductID = od.ProductID

INCORRECT (DO NOT USE THESE FORMATS):
SELECT * FROM dbo.ProductVendor WHERE ProductID = 317
SELECT * FROM ProductVendor WHERE ProductID = 317
SELECT * FROM dbo.Purchasing.ProductVendor WHERE ProductID = 317

DO NOT USE CTEs (WITH clause) or multiple statements. Use simple, single SELECT statements.
"""
        
        # Reference resolution hints based on the follow-up query
        reference_resolution = ""
        if any(word in user_query.lower() for word in ['it', 'them', 'they', 'these', 'those']):
            reference_resolution = f"""
## Reference Resolution
The user's follow-up question contains references that likely refer to:
- The results of the previous query
- The entities mentioned in the previous question: "{previous_query}"
- The tables used in the previous query: {', '.join(previous_tables) if previous_tables else 'None identified'}

Please resolve these references appropriately in your SQL query.
"""

        prompt = f"""
You are an expert SQL query generator. This is a follow-up question to a previous query.

## TABLE FORMAT INSTRUCTIONS - CRITICAL REQUIREMENT
ALWAYS use the format 'SchemaName.TableName' for ALL table references. This is the ONLY valid format.
NEVER use 'dbo.TableName' format - this is INVALID and will cause errors.
NEVER use 'TableName' without schema - this is INVALID and will cause errors.
ALWAYS use the schema name provided in the table references below.

Examples:
{sql_format_examples}

## SQL STYLE REQUIREMENTS - CRITICAL
1. DO NOT use Common Table Expressions (CTEs) with the WITH clause
2. DO NOT use multiple SQL statements separated by semicolons
3. Use only a SINGLE, SIMPLE SELECT statement
4. For complex operations, use subqueries instead of CTEs
5. Keep queries compatible with basic SQL Server capabilities
6. For complex comparisons, consider using UNION or subqueries instead of WITH clauses

## Context from Previous Interaction
Previous Question: "{previous_query}"
Previous SQL Query: 
```sql
{previous_sql}
```

{previous_sql_analysis}

## Available Tables
These are the tables relevant to the user's query (format as SchemaName.TableName):
{table_examples}

{reference_resolution}

## Guidelines
1. This is a follow-up question, so use context from the previous question and query
2. Maintain consistency with previous tables and fields where appropriate
3. MUST format ALL table references as 'SchemaName.TableName'
4. NEVER use 'dbo.TableName' format
5. Use proper table aliases where appropriate
6. Structure the query for optimal performance
7. If adding new tables, ensure they're properly joined with existing tables
8. If the follow-up implies a comparison, use UNION or subqueries rather than CTEs
9. DO NOT use the WITH clause - use subqueries instead

## User's Follow-up Question
"{user_query}"

## Your SQL Query
```sql
"""
        return prompt
    
    @kernel_function
    async def call_llm_to_generate_sql(self, prompt):
        """Call the LLM API to generate a SQL query."""
        # Import here to avoid circular imports
        try:
            import os
            from openai import OpenAI
            from llm_confidence.logprobs_handler import LogprobsHandler
            
            # Try to get API key from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            
            # Create the client with the API key
            if not hasattr(self, 'openai_client'):
                if not api_key:
                    # If no API key in environment, try to import from config
                    try:
                        import config
                        api_key = config.OPENAI_API_KEY
                    except (ImportError, AttributeError):
                        raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or add it to config.py")
                
                self.openai_client = OpenAI(api_key=api_key)
            
            # Initialize the LogprobsHandler
            logprobs_handler = LogprobsHandler()
            
            # Set up API call parameters
            api_params = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an expert SQL query generator for SQL Server. CRITICAL REQUIREMENT: Always use 'SchemaName.TableName' format (like 'Purchasing.ProductVendor') for table references. NEVER use 'dbo.TableName' format as it will cause errors. Write clean, optimized SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            # Try to add logprobs parameters if the model supports it
            try:
                api_params["logprobs"] = True
                api_params["top_logprobs"] = 5
                print("DEBUG - Added logprobs parameters to API call")
            except Exception as e:
                print(f"DEBUG - Error adding logprobs parameters: {e}")
            
            # Make the API call
            response = self.openai_client.chat.completions.create(**api_params)
            
            print(f"DEBUG - OpenAI API response structure: {dir(response)}")
            print(f"DEBUG - OpenAI response choices structure: {dir(response.choices[0])}")
            
            # Extract the SQL query
            sql_query = response.choices[0].message.content
            
            # Calculate confidence score
            confidence_score = 70.0  # Default moderate confidence score
            try:
                # Check if logprobs attribute exists and is not None
                has_logprobs = False
                if hasattr(response.choices[0], 'logprobs'):
                    print(f"DEBUG - Response has logprobs attribute")
                    logprobs_attr = response.choices[0].logprobs
                    if logprobs_attr is not None:
                        print(f"DEBUG - Logprobs attribute is not None: {type(logprobs_attr)}")
                        has_logprobs = True
                        # Check what attributes are available on the logprobs object
                        print(f"DEBUG - Logprobs object structure: {dir(logprobs_attr)}")
                
                if has_logprobs and hasattr(response.choices[0].logprobs, 'content'):
                    print(f"DEBUG - Logprobs has content attribute")
                    logprobs_content = response.choices[0].logprobs.content
                    if logprobs_content:
                        print(f"DEBUG - Logprobs content is not empty")
                        logprobs_formatted = logprobs_handler.format_logprobs(logprobs_content)
                        print(f"DEBUG - Formatted logprobs: {logprobs_formatted[:5] if logprobs_formatted else 'None'}...")  # Print first 5 entries
                        
                        confidence_scores = logprobs_handler.process_logprobs(logprobs_formatted)
                        print(f"DEBUG - Raw confidence scores: {confidence_scores}")
                        
                        # Get the average confidence if multiple scores were returned
                        if confidence_scores:
                            if isinstance(confidence_scores, dict):
                                # If confidence_scores is a dictionary, take the average of values
                                calculated_score = sum(confidence_scores.values()) / len(confidence_scores)
                                print(f"DEBUG - Confidence score from dict average: {calculated_score}")
                                
                                # Only use calculated score if it's meaningful (> 0)
                                if calculated_score > 0:
                                    confidence_score = calculated_score * 100  # Normalize to 0-100 range
                            elif isinstance(confidence_scores, (int, float)):
                                # If confidence_scores is a single value
                                calculated_score = float(confidence_scores)
                                print(f"DEBUG - Confidence score from single value: {calculated_score}")
                                
                                # Only use calculated score if it's meaningful (> 0)
                                if calculated_score > 0:
                                    confidence_score = calculated_score * 100  # Normalize to 0-100 range
                            
                            print(f"DEBUG - Final confidence score: {confidence_score:.2f}%")
                        else:
                            print("DEBUG - No confidence scores returned from process_logprobs, using default")
                    else:
                        print("DEBUG - Logprobs content is empty, using default confidence score")
                else:
                    print("DEBUG - No logprobs content available, using default confidence score of 70%")
            except Exception as e:
                print(f"Error calculating confidence score: {e}")
                import traceback
                print(traceback.format_exc())
            
            return sql_query, confidence_score
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "", 0.0
