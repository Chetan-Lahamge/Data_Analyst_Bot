from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import io
import plotly
import json
import pandas as pd
import asyncio
import uuid
from typing import List, Dict, Any, Optional
import os
import re
import traceback
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import psycopg2
from Agents.observation_generator import generate_observations

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing agents
from Agents.data_catalogue_agent import get_db_connection
from Agents.sql_query_generator_agent import SQLQueryGeneratorAgent
from Agents.data_viz_agent import DataVizAgent
from Agents.data_extractor_agent import DataExtractorAgent

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
import config
from memory.data_wizard_memory import DataWizardMemory

# Create FastAPI app
app = FastAPI(title="Data Wizard API", 
              description="API for natural language to SQL queries with visualizations",
              version="1.0.0")

# Add CORS middleware to allow cross-origin requests from the Angular app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your Angular app's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store sessions (in a real application, use a database)
sessions = {}

# Create a global memory instance
data_wizard_memory = DataWizardMemory()

# Create temp_plots directory if it doesn't exist
os.makedirs(os.path.join(os.getcwd(), 'temp_plots'), exist_ok=True)

# Mount the static directory to serve plot files
app.mount("/temp_plots", StaticFiles(directory="temp_plots"), name="temp_plots")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    useremail: Optional[str] = None
    userid: Optional[str] = None

class ClearRequest(BaseModel):
    session_id: str

class FeedbackQuestion(BaseModel):
    quesid: str
    correctnessValue: Optional[str] = None
    summaryValue: Optional[str] = None
    understandValue: Optional[str] = None
    contentValue: Optional[str] = None
    effortsValue: Optional[str] = None
    biasedValue: Optional[str] = None
    feedbackMessage: Optional[str] = None

class FeedbackRequest(BaseModel):
    chatQues: str
    ques: List[FeedbackQuestion]
    userid: str
    useremail: str

class QueryResponse(BaseModel):
    session_id: str
    sql_query: str
    sql_explanation: Optional[str] = None  # Explanation of SQL query logic
    confidence_score: float
    data: Optional[List[Dict[str, Any]]] = None
    visualization: Optional[str] = None  # Base64 encoded image or JSON
    visualization_explanation: Optional[str] = None  # Explanation of visualization choice
    is_interactive: Optional[bool] = False  # Flag to indicate if the visualization is interactive
    plot_filename: Optional[str] = None  # Path to the temporary plot file
    message: str
    observations: Optional[str] = None  # 🆕 NEW FIELD

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]

def get_session(session_id: str = None):
    """Get or create a session"""
    if session_id and session_id in sessions:
        return sessions[session_id]
    
    # Create a new session
    new_id = session_id or str(uuid.uuid4())
    
    # Initialize agents
    connection = get_db_connection()
    sql_gen_agent = SQLQueryGeneratorAgent()
    data_extractor_agent = DataExtractorAgent(connection=connection)
    data_viz_agent = DataVizAgent(data_extractor_agent)
    
    # Initialize LangChain memory and conversation chain
    langchain_memory = ConversationBufferMemory()
    conversation_chain = ConversationChain(
        llm=ChatOpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY),
        memory=langchain_memory,
        verbose=True
    )
    
    # Create the session with enhanced context tracking
    sessions[new_id] = {
        "session_id": new_id,
        "sql_gen_agent": sql_gen_agent,
        "data_extractor_agent": data_extractor_agent,
        "data_viz_agent": data_viz_agent,
        "langchain_memory": langchain_memory,
        "conversation_chain": conversation_chain,
        "conversation": [],
        # Entity tracking for better context retention
        "mentioned_entities": set(),  # Track mentioned business entities
        "used_tables": set(),         # Track tables used in SQL queries
        "query_metrics": set(),       # Track metrics mentioned (sales, revenue, etc)
        "last_query_context": {       # Detailed context about most recent query
            "sql": None,
            "tables": [],
            "columns": [],
            "filters": [],
            "result_summary": None,
            "visualization_type": None
        },
        # Tracking previous visualizations 
        "visualizations": [],
        # Context window management
        "context_window_size": 10     # Maximum turns to keep in active memory
    }
    
    return sessions[new_id]

def figure_to_base64(fig):
    """Convert a figure to appropriate format for transmission"""
    # For Plotly figures
    if hasattr(fig, 'to_dict'):
        try:
            import json
            import plotly
            
            # Get the figure as JSON
            fig_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
            print(f"Successfully converted Plotly figure to JSON (length: {len(fig_json)})")
            return fig_json
        except Exception as e:
            print(f"Error converting Plotly figure to JSON: {e}")
            traceback.print_exc()
            return None
    
    # Fallback for matplotlib figures
    try:
        import matplotlib.pyplot as plt
        import io
        import base64
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)  # Close the figure to free memory
        return img_str
    except Exception as e:
        print(f"Error converting matplotlib figure to PNG: {e}")
        return None

def is_visualization_only_request(query):
    """Check if the query is only asking to change visualization"""
    viz_keywords = [
        "change to", "convert to", "switch to", "show as", "display as",
        "line chart", "bar chart", "pie chart", "scatter plot", 
        "histogram", "graph", "plot"
    ]
    
    # Check if query is short and contains visualization keywords
    if len(query.split()) < 15:  # Short query
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in viz_keywords)
    
    return False

async def process_query(query):
    try:
        # Initialize default values
        response_text = ""
        generated_query = ""
        query_results = pd.DataFrame()
        viz_data = None
        is_interactive = False
        sql_explanation = ""
        viz_explanation = ""
        
        # Check if this is a visualization-only request
        visualization_request = is_visualization_only_request(query)
        logger.info(f"Is visualization-only request: {visualization_request}")
        
        if visualization_request and 'data_wizard_memory' in globals() and data_wizard_memory.has_data:
            logger.info(f"Using stored data from memory for visualization request")
            
            # Skip SQL generation and use stored data
            response_text = "I've updated the visualization based on your request."
            generated_query = data_wizard_memory.previous_sql
            query_results = data_wizard_memory.latest_dataframe
            sql_explanation = data_wizard_memory.sql_explanation
            
            # Determine the requested visualization type
            viz_type = None
            if "line" in query.lower():
                viz_type = "line"
            elif "bar" in query.lower():
                viz_type = "bar"
            elif "pie" in query.lower():
                viz_type = "pie"
            elif "scatter" in query.lower():
                viz_type = "scatter"
            
            logger.info(f"Requested visualization type: {viz_type}")
            
            # Generate new visualization with the stored data
            try:
                # Create a new modified query that combines the original question with the viz request
                modified_query = f"{data_wizard_memory.original_question} (Showing as {viz_type if viz_type else 'requested'} chart)"
                
                # In your data_viz_agent.py, add a parameter to force visualization type
                fig_result = await data_viz_agent.determine_plot_type(
                    query_results, 
                    modified_query,
                    viz_type  # Pass the visualization type
                )
                
                if fig_result and len(fig_result) >= 2:
                    fig, viz_explanation = fig_result
                    viz_data = figure_to_base64(fig)
                    is_interactive = hasattr(fig, 'to_html') or hasattr(fig, 'to_dict')
                else:
                    viz_data = None
                    viz_explanation = "Could not generate visualization"
                    is_interactive = False
            except Exception as viz_error:
                logger.error(f"Error generating visualization: {str(viz_error)}")
                viz_data = None
                viz_explanation = f"Error generating visualization: {str(viz_error)}"
                is_interactive = False
        else:
            # Begin with conversation chain
            response_text = await conversation_chain.predict(input=query)
            
            # Try to generate SQL query
            logger.info(f"Starting SQL query generation for: {query}")
            generated_query = await sql_gen_agent.generate_sql_query(
                user_query=query,
                conversation_history=conversation
            )
            
            # Clean the query - remove markdown formatting if present
            if generated_query:
                # Remove markdown formatting if present (```sql and ```)
                if "```sql" in generated_query:
                    sql_parts = generated_query.split("```sql")
                    if len(sql_parts) > 1:
                        sql_content = sql_parts[1].split("```")[0].strip()
                        generated_query = sql_content
                
                # Remove SQL comments (lines starting with --)
                cleaned_lines = []
                for line in generated_query.split('\n'):
                    if not line.strip().startswith('--'):
                        cleaned_lines.append(line)
                generated_query = '\n'.join(cleaned_lines).strip()
                
                logger.info(f"Cleaned query for execution: {generated_query}")
                
                # Execute the query if one was generated
                try:
                    df = await data_extractor_agent.execute_sql_query(generated_query)
                    query_results = df if df is not None else pd.DataFrame()
                except Exception as sql_error:
                    logger.error(f"Error executing SQL query: {str(sql_error)}")
                    query_results = pd.DataFrame()
            
            # Generate SQL explanation
            if generated_query:
                sql_explanation_prompt = f"""
                The user asked: "{query}"
                
                You generated this SQL query to answer their question:
                ```sql
                {generated_query}
                ```
                
                Please provide a brief, clear explanation (2-3 sentences) of:
                1. Why you selected these specific tables and columns
                2. How the query's logic answers the user's question
                3. What the results will tell the user
                
                Keep your explanation concise and in plain language that a business user would understand.
                """
                
                # Get SQL explanation
                sql_explanation = await conversation_chain.predict(input=sql_explanation_prompt)
            
            # Generate visualization if we have data
            if not query_results.empty:
                try:
                    fig_result = await data_viz_agent.determine_plot_type(query_results, query)
                    
                    if fig_result and len(fig_result) >= 2:
                        fig, viz_explanation = fig_result
                        viz_data = figure_to_base64(fig)
                        is_interactive = hasattr(fig, 'to_html') or hasattr(fig, 'to_dict')
                    else:
                        viz_data = None
                        viz_explanation = ""
                        is_interactive = False
                except Exception as viz_error:
                    logger.error(f"Error generating visualization: {str(viz_error)}")
                    viz_data = None
                    viz_explanation = ""
                    is_interactive = False
        
        # Store data in memory if available
        try:
            if 'data_wizard_memory' in globals():
                if query_results is not None and not query_results.empty:
                    data_wizard_memory.latest_dataframe = query_results
                    data_wizard_memory.previous_sql = generated_query
                    data_wizard_memory.sql_explanation = sql_explanation
                    data_wizard_memory.original_question = query
                    data_wizard_memory.visualization_explanation = viz_explanation
                    data_wizard_memory.is_interactive = is_interactive
        except Exception as mem_error:
            logger.error(f"Error updating memory: {str(mem_error)}")
        
        # Return the response
        return {
            "message": response_text,
            "data": query_results.to_dict(orient='records') if not query_results.empty else [],
            "sql_query": generated_query,
            "visualization": viz_data,
            "is_interactive": is_interactive,
            "sql_explanation": sql_explanation,
            "visualization_explanation": viz_explanation
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "message": f"I encountered an error: {str(e)}",
            "data": [],
            "sql_query": "",
            "visualization": None,
            "is_interactive": False
        }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    session = get_session(request.session_id)
    session_id = session["session_id"]
    
    try:
        # Add user query to conversation history first
        session["conversation"].append({
            "role": "user", 
            "content": request.query,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Add this to check if memory is initialized
        logger.info(f"Memory object initialized: {data_wizard_memory is not None}")
        
        # Store the original query - add try/except
        try:
            data_wizard_memory.original_question = request.query
        except Exception as e:
            logger.error(f"Error storing question in memory: {str(e)}")
        
        # Check if this is a visualization-only request
        if is_visualization_only_request(request.query) and data_wizard_memory.has_data:
            logger.info(f"Detected visualization-only request: {request.query}")
            
            # Skip SQL generation and use stored data
            response_text = "I've updated the visualization based on your request."
            generated_query = data_wizard_memory.previous_sql
            query_results = data_wizard_memory.latest_dataframe
            sql_explanation = data_wizard_memory.sql_explanation
            
            # Rest of visualization processing remains the same...
        else:
            # Begin with conversation chain for non-visualization requests
            response_text = session["conversation_chain"].predict(input=request.query)
            
            # Normal SQL query generation and execution
            logger.info(f"Starting SQL query generation for: {request.query}")
            sql_result = await session["sql_gen_agent"].generate_sql_query(
                user_query=request.query,
                conversation_history=session["conversation"]  # Important: Pass the conversation history
            )
            
            # Extract SQL query and confidence score
            if isinstance(sql_result, dict) and 'sql_query' in sql_result:
                sql_query = sql_result['sql_query']
                confidence_score = sql_result.get('confidence_score', 70.0)  # Default if not available
                
                # Track tables mentioned in the result for context
                if 'relevant_tables' in sql_result and sql_result['relevant_tables']:
                    for table in sql_result['relevant_tables']:
                        session["used_tables"].add(table)
            else:
                sql_query = str(sql_result)
                confidence_score = 70.0
            
            # Generate SQL explanation
            sql_explanation_prompt = f"""
            The user asked: "{request.query}"
            
            You generated this SQL query to answer their question:
            ```sql
            {sql_query}
            ```
            
            Please provide a brief, clear explanation (2-3 sentences) of:
            1. Why you selected these specific tables and columns
            2. How the query's logic answers the user's question
            3. What the results will tell the user
            
            Keep your explanation concise and in plain language that a business user would understand.
            """
            
            # Get SQL explanation
            sql_explanation_response = session["conversation_chain"].predict(input=sql_explanation_prompt)
            
            # Parse SQL query to extract tables, columns, and filters for context tracking
            # Extract tables
            table_pattern = r'FROM\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)|JOIN\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)'
            table_matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
            tables_used = []
            for match in table_matches:
                for table in match:
                    if table and '.' in table:
                        schema_table = table.strip()
                        if schema_table not in tables_used:
                            tables_used.append(schema_table)
                            # Extract just table name without schema
                            table_name = schema_table.split('.')[1]
                            session["used_tables"].add(table_name)
            
            # Extract columns
            columns_pattern = r'SELECT\s+(.*?)\s+FROM'
            columns_match = re.search(columns_pattern, sql_query, re.IGNORECASE | re.DOTALL)
            columns_used = []
            if columns_match:
                columns_text = columns_match.group(1)
                # Basic column extraction (this is simplified)
                columns_used = [col.strip() for col in columns_text.split(',')]
            
            # Extract filters
            where_pattern = r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|HAVING|$)'
            where_match = re.search(where_pattern, sql_query, re.IGNORECASE | re.DOTALL)
            filters_used = []
            if where_match:
                filters_text = where_match.group(1).strip()
                # Simple filter extraction, can be enhanced
                filters_used = [filters_text]
            
            # Update the last query context for reference in future queries
            session["last_query_context"]["sql"] = sql_query
            session["last_query_context"]["tables"] = tables_used
            session["last_query_context"]["columns"] = columns_used
            session["last_query_context"]["filters"] = filters_used
            
            # Update memory with generated SQL
            session["langchain_memory"].save_context(
                {"input": f"SQL Query for: {request.query}"}, 
                {"output": f"SQL: {sql_query}"}
            )
            
            # Add SQL query to conversation history
            session["conversation"].append({
                "role": "assistant", 
                "content": f"Generated SQL Query:\n```sql\n{sql_query}\n```",
                "confidence_score": confidence_score,
                "timestamp": pd.Timestamp.now().isoformat()
            })
            
            # Execute SQL query
            df = await session["data_viz_agent"].execute_sql_query(sql_query)
            
            # Default response with SQL only
            response = {
                "session_id": session_id,
                "sql_query": sql_query,
                "sql_explanation": sql_explanation_response,
                "confidence_score": confidence_score,
                "message": "SQL query generated successfully.",
                "is_interactive": False
            }
            
            if df is not None and not df.empty:
                # Convert DataFrame to dict for JSON response
                # Limit to max 50 rows to prevent very large responses
                MAX_ROWS = 50
                if len(df) > MAX_ROWS:
                    display_df = df.head(MAX_ROWS)
                    response["message"] = f"Query returned {len(df)} rows (showing first {MAX_ROWS})."
                else:
                    display_df = df
                    response["message"] = f"Query returned {len(df)} rows."
                
                # Convert to dict for JSON response
                data_records = display_df.to_dict(orient='records')
                response["data"] = data_records
                
                # Generate a result summary for context tracking
                result_summary = f"Query returned {len(df)} rows with columns: {', '.join(df.columns)}"
                
                # Track key stats from the result (min, max, avg for numeric columns)
                stats_summary = []
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats_summary.append(f"{col} range: {df[col].min()} to {df[col].max()}, avg: {df[col].mean():.2f}")
                
                if stats_summary:
                    result_summary += "\nKey statistics: " + "; ".join(stats_summary)
                
                # Update the last query context with result summary
                session["last_query_context"]["result_summary"] = result_summary
                
                # Update memory with results summary
                session["langchain_memory"].save_context(
                    {"input": "Query results"}, 
                    {"output": result_summary}
                )
                
                # Add results to conversation with timestamp
                session["conversation"].append({
                    "role": "assistant", 
                    "content": "Here are the query results:", 
                    "dataframe": df.to_dict(orient='records'),
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                
                # Generate visualization
                fig_result = await session["data_viz_agent"].determine_plot_type(df, sql_query)
                
                # Now fig_result is a tuple (fig, explanation)
                if fig_result and fig_result[0]:
                    fig, viz_explanation = fig_result
                    
                    # Check if this is a Plotly figure
                    is_plotly = hasattr(fig, 'to_html') or hasattr(fig, 'to_dict')
                    
                    # Update the last query context with visualization type
                    session["last_query_context"]["visualization_type"] = "interactive" if is_plotly else "static"
                    
                    # Save the plot to a temporary file if it's a Plotly figure
                    plot_filename = None
                    if is_plotly:
                        try:
                            plot_filename = session["data_viz_agent"].save_plot_to_temp_file(fig)
                            print(f"Plot saved to temporary file: {plot_filename}")
                        except Exception as e:
                            logger.error(f"Error saving plot to temp file: {str(e)}")
                            traceback.print_exc()
                    
                    # Convert plot to appropriate format for backwards compatibility
                    viz_data = figure_to_base64(fig)
                    if viz_data:
                        print(f"Visualization data type: {type(viz_data)}, Interactive: {is_plotly}")
                        response["visualization"] = viz_data
                        response["visualization_explanation"] = viz_explanation
                        response["is_interactive"] = is_plotly
                        
                        # Add plot filename to response if available
                        if plot_filename:
                            response["plot_filename"] = plot_filename
                        
                        # Store visualization in history (limited to last 5)
                        viz_entry = {
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "type": "interactive" if is_plotly else "static",
                            "explanation": viz_explanation,
                            "related_query": request.query
                        }
                        session["visualizations"].insert(0, viz_entry)
                        if len(session["visualizations"]) > 5:
                            session["visualizations"] = session["visualizations"][:5]
                        
                        # Update memory with visualization info
                        session["langchain_memory"].save_context(
                            {"input": "Visualization created"}, 
                            {"output": f"A {'interactive' if is_plotly else 'static'} visualization was created for the query results"}
                        )
                        
                        # Add visualization to conversation
                        session["conversation"].append({
                            "role": "assistant", 
                            "content": f"Here's the visualization: {viz_explanation}",
                            "figure": True,  # Just mark that a figure exists, we don't store it in the history
                            "timestamp": pd.Timestamp.now().isoformat()
                        })
            else:
                response["message"] = "No data returned from the query."
                session["conversation"].append({
                    "role": "assistant", 
                    "content": "No data returned from the query.",
                    "timestamp": pd.Timestamp.now().isoformat()
                })
            
            # Manage context window - keep only the most recent N conversation turns
            # But always keep the first turn as it often contains important initial context
            if len(session["conversation"]) > (session["context_window_size"] * 2 + 1):
                # Keep first message plus most recent messages
                session["conversation"] = [
                    session["conversation"][0], 
                    *session["conversation"][-(session["context_window_size"]*2):]
                ]
            
            # Store the query results
            data_wizard_memory.latest_dataframe = df
            # Generate observations for the table and chart
            if df is not None and not df.empty:
                observations = await generate_observations(df, request.query)
                response["observations"] = observations
            else:
                response["observations"] = "No observations available since the data is empty."
            
            # Store the query explanation
            data_wizard_memory.sql_explanation = sql_explanation_response
            
            # Initialize is_plotly variable with a default value
            is_plotly = False

            # Only set it if we have a visualization
            if "visualization" in response and response["visualization"] and "is_interactive" in response:
                is_plotly = response["is_interactive"]

            data_wizard_memory.visualization_type = "interactive" if is_plotly else "static"
            data_wizard_memory.visualization_explanation = viz_explanation if "viz_explanation" in locals() else ""
            data_wizard_memory.is_interactive = is_plotly
            
            # Add this line right before the return statement
            print(f"Memory after query: {data_wizard_memory}")
            
            # Store the chat interaction in the database
            background_tasks.add_task(
                store_chat_interaction,
                request.useremail,          # Changed from email_id
                request.userid,             # Changed from user_id
                request.query,              # User's query
                response["message"],        # Bot's response text
                response.get("sql_query", "") # Generated SQL query
            )
            
            return response
        
    except Exception as e:
        # Add error to conversation history
        session["conversation"].append({
            "role": "assistant", 
            "content": f"An error occurred: {str(e)}",
            "timestamp": pd.Timestamp.now().isoformat()
        })
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())  # This gives the full stack trace
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Convert conversation history to a format suitable for JSON
    # Remove any non-serializable objects like matplotlib figures
    history = []
    for message in session["conversation"]:
        msg_copy = message.copy()
        if "dataframe" in msg_copy:
            # Convert DataFrame to dict if it's a pandas DataFrame
            if isinstance(msg_copy["dataframe"], pd.DataFrame):
                msg_copy["dataframe"] = msg_copy["dataframe"].to_dict(orient='records')
        if "figure" in msg_copy:
            # Just indicate a figure exists but don't include the actual figure
            msg_copy["figure"] = True
        history.append(msg_copy)
    
    return {"session_id": session_id, "history": history}

@app.post("/api/clear")
async def clear_history(request: ClearRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove the session
    del sessions[request.session_id]
    
    return {"status": "success", "message": "Session cleared successfully"}

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        # Step 1: Get the latest entry from experience_ai_chat for this user and query
        connection = get_pgsql_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Failed to connect to database")
            
        cursor = connection.cursor()
        
        # SQL to find the latest chat entry for this user and query
        find_chat_sql = """
        SELECT chat_id, chat_ques 
        FROM experience_ai_chat 
        WHERE user_id = %s AND chat_ques = %s
        ORDER BY chat_date DESC
        LIMIT 1
        """
        
        cursor.execute(find_chat_sql, (request.userid, request.chatQues))
        chat_row = cursor.fetchone()
        
        if not chat_row:
            connection.close()
            raise HTTPException(status_code=404, detail="Chat entry not found")
            
        chat_id = chat_row[0]
        chat_ques = chat_row[1]
        
        # Step 2: Insert feedback rows into experienceai_feedback
        insert_feedback_sql = """
        INSERT INTO experienceai_feedback
        (ques_id, chat_quesid, feedback_ans, feedback_date, chatques, user_id)
        VALUES (%s, %s, %s, NOW(), %s, %s)
        """
        
        # Insert each feedback question response
        for question in request.ques:
            # Determine the feedback answer based on the question id
            feedback_ans = None
            if question.quesid == "1" and question.correctnessValue:
                feedback_ans = question.correctnessValue
            elif question.quesid == "2" and question.summaryValue:
                feedback_ans = question.summaryValue
            elif question.quesid == "3" and question.understandValue:
                feedback_ans = question.understandValue
            elif question.quesid == "4" and question.contentValue:
                feedback_ans = question.contentValue
            elif question.quesid == "5" and question.effortsValue:
                feedback_ans = question.effortsValue
            elif question.quesid == "6" and question.biasedValue:
                feedback_ans = question.biasedValue
            elif question.quesid == "7" and question.feedbackMessage:
                feedback_ans = question.feedbackMessage
                
            # Skip if no answer
            if not feedback_ans:
                continue
                
            # Insert the feedback
            cursor.execute(insert_feedback_sql, (
                question.quesid,   # ques_id
                chat_id,           # chat_quesid
                feedback_ans,      # feedback_ans
                chat_ques,         # chatques
                request.userid     # user_id
            ))
        
        # Commit all the inserts
        connection.commit()
        cursor.close()
        connection.close()
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Data Wizard API is running. Visit /docs for API documentation."}

# Dashboard endpoints added from app.py
@app.get("/api/users")
async def get_users():
    """Get all users from the database"""
    try:
        connection = get_pgsql_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Failed to connect to database")
            
        cursor = connection.cursor()
        query = "SELECT * FROM users"  # Changed from user_master to users
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to list of dicts
        users = []
        for row in rows:
            users.append(dict(zip(columns, row)))
            
        cursor.close()
        connection.close()
        
        return users
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/getFeedback/{user_id}")
@app.get("/api/getFeedback")
async def get_feedback(user_id: Optional[int] = None):
    """Get feedback data with optional user_id filter"""
    try:
        connection = get_pgsql_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Failed to connect to database")
            
        cursor = connection.cursor()
        
        # Build query based on whether user_id is provided
        if user_id:
            query = f"""
            SELECT 
                a.chat_id, 
                a.chat_ques, 
                a.chat_answer as chatanswer, 
                TO_CHAR(a.chat_date, 'Dy, DD Mon YYYY HH24:MI:SS') || ' GMT' as chat_date,
                a.user_id,
                u.user_name,
                b.chat_quesid as chatquesid,
                b.ques_id,
                b.feedback_ans as feedbackans,
                TO_CHAR(b.feedback_date, 'Dy, DD Mon YYYY HH24:MI:SS') || ' GMT' as feedbackdate,
                b.ques_id as feedbackquesid
            FROM experience_ai_chat a
            LEFT JOIN experienceai_feedback b ON a.chat_id = b.chat_quesid
            LEFT JOIN users u ON a.user_id = u.user_id
            WHERE a.user_id = {user_id}
            """
        else:
            query = """
            SELECT 
                a.chat_id, 
                a.chat_ques, 
                a.chat_answer as chatanswer, 
                TO_CHAR(a.chat_date, 'Dy, DD Mon YYYY HH24:MI:SS') || ' GMT' as chat_date,
                a.user_id,
                u.user_name,
                b.chat_quesid as chatquesid,
                b.ques_id,
                b.feedback_ans as feedbackans,
                TO_CHAR(b.feedback_date, 'Dy, DD Mon YYYY HH24:MI:SS') || ' GMT' as feedbackdate,
                b.ques_id as feedbackquesid
            FROM experience_ai_chat a
            LEFT JOIN experienceai_feedback b ON a.chat_id = b.chat_quesid
            LEFT JOIN users u ON a.user_id = u.user_id
            """
            
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to list of dicts
        chats = []
        for row in rows:
            chats.append(dict(zip(columns, row)))
            
        cursor.close()
        connection.close()
        
        return chats
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def get_pgsql_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            host='0.0.0.0',
            dbname='analyticsbot',
            user='analyticsbot',
            password='analyticsbot@123',
            port='5432'
        )
        return connection
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        return None

def store_chat_interaction(email_id=None, user_id=None, user_query="", bot_response="", sql_query=""):
    """Store chat interaction in PostgreSQL database with the correct columns"""
    try:
        # Skip storing if email_id or user_id is missing
        if not email_id or not user_id:
            logger.warning("Missing email_id or user_id - chat interaction not stored")
            return False
        
        # Get database connection
        connection = get_pgsql_db_connection()
        if not connection:
            logger.error("Failed to connect to PostgreSQL database")
            return False
            
        cursor = connection.cursor()
        
        # SQL to insert chat interaction with the correct column names
        insert_sql = """
        INSERT INTO experience_ai_chat 
        (chat_ques, chat_answer, chat_date, user_id, user_email) 
        VALUES (%s, %s, NOW(), %s, %s)
        """
        
        # Execute query using ONLY values from the UI request
        cursor.execute(insert_sql, (
            user_query,       # chat_ques: what the user asked
            sql_query,        # chat_answer: the SQL that was generated
            user_id,          # user_id: from request
            email_id          # user_email: from request
        ))
        connection.commit()
        
        # Close resources
        cursor.close()
        connection.close()
        
        logger.info(f"Stored chat interaction for user: {email_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing chat interaction: {str(e)}")
        return False

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
