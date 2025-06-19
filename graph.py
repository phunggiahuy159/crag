# # from langgraph.graph import END, StateGraph
# # from typing_extensions import TypedDict
# # from typing import List
# # from nodes.transform_node import transform_query
# # from nodes.retrieve_node import retrieve
# # from nodes.search_node import web_search
# # from nodes.grade_node import grade_documents
# # from nodes.generate_node import generate
# # from nodes.decision_node import decide_to_generate


# # def workflow_compiler():
# #     class GraphState(TypedDict):
# #         question: str
# #         generation: str
# #         web_search: str
# #         documents: List[str]

# #     workflow = StateGraph(GraphState)

# #     workflow.add_node("retrieve", retrieve)
# #     workflow.add_node("grade_documents", grade_documents)
# #     workflow.add_node("generate", generate)
# #     workflow.add_node("transform_query", transform_query)
# #     workflow.add_node("web_search_node", web_search)

# #     workflow.set_entry_point("retrieve")
# #     workflow.add_edge("retrieve", "grade_documents")
# #     workflow.add_conditional_edges(
# #         "grade_documents",
# #         decide_to_generate,
# #         {
# #             "transform_query": "transform_query",
# #             "generate": "generate",
# #         },
# #     )
# # from langgraph.graph import END, StateGraph
# # from typing_extensions import TypedDict
# # from typing import List, Literal
# # from langchain_core.messages import AIMessage
# # from langchain_community.utilities import SQLDatabase
# # from langchain_community.agent_toolkits import SQLDatabaseToolkit
# # from langgraph.prebuilt import ToolNode
# # from langchain.chat_models import init_chat_model

# # # Import your existing nodes
# # from nodes.transform_node import transform_query
# # from nodes.retrieve_node import retrieve
# # from nodes.search_node import web_search
# # from nodes.grade_node import grade_documents
# # from nodes.generate_node import generate
# # from nodes.decision_node import decide_to_generate


# # class HybridGraphState(TypedDict):
# #     question: str
# #     generation: str
# #     web_search: str
# #     documents: List[str]
# #     query_type: str  # "sql" or "crag"
# #     sql_result: str
# #     sql_query: str
# #     table_schemas: str


# # def classify_query_type(state: HybridGraphState) -> HybridGraphState:
# #     """
# #     Classify whether the question should be handled by SQL or CRAG
# #     """
# #     question = state["question"]
    
# #     # SQL indicators - customize these based on your database schema
# #     sql_keywords = [
# #         # Data aggregation keywords
# #         "how many", "count", "total", "sum", "average", "maximum", "minimum",
# #         "most", "least", "highest", "lowest", "top", "bottom",
        
# #         # Time-based queries
# #         "in 2023", "in 2024", "last month", "this year", "between",
# #         "sales", "revenue", "orders", "customers", "products",
        
# #         # Comparison queries  
# #         "compare", "versus", "vs", "difference between",
        
# #         # Specific data retrieval
# #         "list all", "show me", "get all", "find records",
        
# #         # Add your specific database entities
# #         "employees", "invoices", "artists", "albums", "tracks",'trong cơ sở dữ liệu'
# #     ]
    
# #     # CRAG indicators - conceptual/analytical questions
# #     crag_keywords = [
# #         "explain", "what is", "how does", "why", "concept", "definition",
# #         "meaning", "understand", "learn about", "tell me about",
# #         "describe", "analysis", "insights", "trends", "patterns",
# #         "recommendations", "best practices", "methodology"
# #     ]
    
# #     question_lower = question.lower()
    
# #     # Score based on keyword presence
# #     sql_score = sum(1 for keyword in sql_keywords if keyword in question_lower)
# #     crag_score = sum(1 for keyword in crag_keywords if keyword in question_lower)
    
# #     # Decision logic
# #     if sql_score > crag_score and sql_score > 0:
# #         query_type = "sql"
# #     elif crag_score > 0:
# #         query_type = "crag"
# #     else:
# #         # Default fallback - could be made smarter with ML classification
# #         if any(char.isdigit() for char in question) or "?" in question:
# #             query_type = "sql"
# #         else:
# #             query_type = "crag"
    
# #     return {"query_type": query_type}


# # def route_query(state: HybridGraphState) -> Literal["sql_pipeline", "crag_pipeline"]:
# #     """
# #     Route to either SQL or CRAG pipeline based on classification
# #     """
# #     return "sql_pipeline" if state["query_type"] == "sql" else "crag_pipeline"


# # # SQL Pipeline Nodes
# # def get_sql_schema(state: HybridGraphState, db: SQLDatabase) -> HybridGraphState:
# #     """
# #     Get relevant database schema information
# #     """
# #     # Get all table names
# #     tables = db.get_usable_table_names()
    
# #     # Get schema for relevant tables (you can make this smarter)
# #     schema_info = []
# #     for table in tables:  # Limit to avoid token overflow
# #         try:
# #             schema = db.get_table_info([table])
# #             schema_info.append(schema)
# #         except Exception as e:
# #             continue
    
# #     return {"table_schemas": "\n".join(schema_info)}


# # def generate_sql_query(state: HybridGraphState, llm) -> HybridGraphState:
# #     """
# #     Generate SQL query based on the question and schema
# #     """
# #     system_prompt = f"""
# #     You are a SQL expert. Given the database schema and user question, 
# #     generate a syntactically correct SQL query.
    
# #     Database Schema:
# #     {state.get("table_schemas", "")}
    
# #     Rules:
# #     - Only use SELECT statements
# #     - Limit results to 10 unless specified otherwise
# #     - Use proper JOIN syntax when needed
# #     - Handle NULL values appropriately
    
# #     Question: {state["question"]}
    
# #     Return only the SQL query, no explanations.
# #     """
    
# #     response = llm.invoke(system_prompt)
# #     sql_query = response.content.strip()
    
# #     # Clean up the query (remove markdown formatting if present)
# #     if "```sql" in sql_query:
# #         sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
# #     elif "```" in sql_query:
# #         sql_query = sql_query.split("```")[1].strip()
    
# #     return {"sql_query": sql_query}


# # def execute_sql_query(state: HybridGraphState, db: SQLDatabase) -> HybridGraphState:
# #     """
# #     Execute the generated SQL query
# #     """
# #     try:
# #         result = db.run(state["sql_query"])
# #         return {"sql_result": str(result)}
# #     except Exception as e:
# #         # If query fails, could retry with correction or fall back to CRAG
# #         return {"sql_result": f"Error executing query: {str(e)}"}


# # def generate_sql_response(state: HybridGraphState, llm) -> HybridGraphState:
# #     """
# #     Generate natural language response from SQL results
# #     """
# #     prompt = f"""
# #     Based on the SQL query results, provide a clear and concise answer to the user's question.
    
# #     Question: {state["question"]}
# #     SQL Query: {state["sql_query"]}
# #     Results: {state["sql_result"]}
    
# #     Provide a natural language answer based on these results.
# #     """
    
# #     response = llm.invoke(prompt)
# #     return {"generation": response.content}


# # def workflow_compiler(db: SQLDatabase, llm):
# #     """
# #     Compile the hybrid workflow that can handle both SQL and CRAG queries
# #     """
    
# #     workflow = StateGraph(HybridGraphState)
    
# #     # Add classification node
# #     workflow.add_node("classify_query", classify_query_type)
    
# #     # SQL Pipeline nodes
# #     workflow.add_node("get_sql_schema", lambda state: get_sql_schema(state, db))
# #     workflow.add_node("generate_sql_query", lambda state: generate_sql_query(state, llm))
# #     workflow.add_node("execute_sql_query", lambda state: execute_sql_query(state, db))
# #     workflow.add_node("generate_sql_response", lambda state: generate_sql_response(state, llm))
    
# #     # CRAG Pipeline nodes (your existing nodes)
# #     workflow.add_node("retrieve", retrieve)
# #     workflow.add_node("grade_documents", grade_documents)
# #     workflow.add_node("generate", generate)
# #     workflow.add_node("transform_query", transform_query)
# #     workflow.add_node("web_search_node", web_search)
    
# #     # Set entry point
# #     workflow.set_entry_point("classify_query")
    
# #     # Add conditional routing after classification
# #     workflow.add_conditional_edges(
# #         "classify_query",
# #         route_query,
# #         {
# #             "sql_pipeline": "get_sql_schema",
# #             "crag_pipeline": "retrieve",
# #         },
# #     )
    
# #     # SQL Pipeline edges
# #     workflow.add_edge("get_sql_schema", "generate_sql_query")
# #     workflow.add_edge("generate_sql_query", "execute_sql_query")
# #     workflow.add_edge("execute_sql_query", "generate_sql_response")
# #     workflow.add_edge("generate_sql_response", END)
    
# #     # CRAG Pipeline edges (your existing flow)
# #     workflow.add_edge("retrieve", "grade_documents")
# #     workflow.add_conditional_edges(
# #         "grade_documents",
# #         decide_to_generate,
# #         {
# #             "transform_query": "transform_query",
# #             "generate": "generate",
# #         },
# #     )
# #     workflow.add_edge("transform_query", "web_search_node")
# #     workflow.add_edge("web_search_node", "generate")
# #     workflow.add_edge("generate", END)
    
# #     return workflow.compile()


# # # Usage example
# # def create_hybrid_system(database_uri: str, llm):
# #     """
# #     Create the complete hybrid system
# #     """
# #     # Initialize database
# #     db = SQLDatabase.from_uri(database_uri)
    
# #     # Compile workflow
# #     workflow = workflow_compiler(db, llm)
    
# #     return workflow


# # # Example usage:

# # # Initialize your LLM and database
# # # llm = init_chat_model("openai:gpt-4")
# # from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# # import streamlit as st

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.0-flash", google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
# # )
# # url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
# # import requests

# # response = requests.get(url)

# # if response.status_code == 200:
# #     # Open a local file in binary write mode
# #     with open("Chinook.db", "wb") as file:
# #         # Write the content of the response (the file) to the local file
# #         file.write(response.content)
# #     print("File downloaded and saved as Chinook.db")
# # else:
# #     print(f"Failed to download the file. Status code: {response.status_code}")
# # db_uri = "sqlite:///Chinook.db"

# # # Create hybrid workflow
# # hybrid_workflow = create_hybrid_system(db_uri, llm)

# # # Test with different query types
# # sql_question = "Which sales agent made the most in sales in 2009?"
# # crag_question = "What are the best practices for customer retention?"

# # # Run SQL query
# # # result1 = hybrid_workflow.invoke({"question": sql_question})
# # # print(result1)
# # # Run CRAG query  
# # # result2 = hybrid_workflow.invoke({"question": crag_question})
# from langgraph.graph import END, StateGraph
# from langgraph.prebuilt import create_react_agent
# from typing_extensions import TypedDict
# from typing import List, Literal
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langgraph.prebuilt import ToolNode
# from langchain.chat_models import init_chat_model

# # Import your existing nodes
# from nodes.transform_node import transform_query
# from nodes.retrieve_node import retrieve
# from nodes.search_node import web_search
# from nodes.grade_node import grade_documents
# from nodes.generate_node import generate
# from nodes.decision_node import decide_to_generate


# class HybridGraphState(TypedDict):
#     question: str
#     generation: str
#     web_search: str
#     documents: List[str]
#     query_type: str  # "sql" or "crag"
#     sql_result: str
#     sql_query: str
#     table_schemas: str
#     messages: List  # For React agent communication
#     agent_outcome: str


# def classify_query_type(state: HybridGraphState) -> HybridGraphState:
#     """
#     Enhanced query classification with better heuristics
#     """
#     question = state["question"]
    
#     # SQL indicators - more comprehensive list
#     sql_keywords = [
#         # Data aggregation keywords,
#         'In my database',
#         "how many", "count", "total", "sum", "average", "mean", "maximum", "minimum",
#         "most", "least", "highest", "lowest", "top", "bottom", "rank", "sort",
        
#         # Time-based queries
#         "in 2023", "in 2024", "last month", "this year", "between", "since", "before", "after",
#         "during", "quarterly", "monthly", "yearly", "daily",
        
#         # Business metrics
#         "sales", "revenue", "profit", "orders", "customers", "products", "inventory",
#         "transactions", "purchases", "invoices", "employees", "performance",
        
#         # Comparison queries  
#         "compare", "versus", "vs", "difference between", "better than", "worse than",
        
#         # Specific data retrieval
#         "list all", "show me", "get all", "find records", "search for", "lookup",
#         "who are", "what are", "which", "where are",
        
#         # Database-specific entities (customize based on your schema)
#         "artists", "albums", "tracks", "genres", "playlists", "media types",
#         "invoice", "customer", "employee", "support rep",'nhân viên','sản phẩm','chi nhánh','bán hàng'
#     ]
    
#     # CRAG indicators - conceptual/analytical questions
#     crag_keywords = [
#         "explain", "what is", "how does", "why", "concept", "definition",
#         "meaning", "understand", "learn about", "tell me about",
#         "describe", "analysis", "insights", "trends", "patterns",
#         "recommendations", "best practices", "methodology", "strategy",
#         "approach", "theory", "principle", "guide", "tutorial"
#     ]
    
#     question_lower = question.lower()
    
#     # Enhanced scoring with weighted keywords
#     sql_score = 0
#     crag_score = 0
    
#     # Weight important SQL indicators higher
#     high_value_sql = ["how many", "count", "total", "sum", "average", "list all", "top", "most"]
#     high_value_crag = ["explain", "what is", "best practices", "methodology"]
    
#     for keyword in sql_keywords:
#         if keyword in question_lower:
#             weight = 3 if keyword in high_value_sql else 1
#             sql_score += weight
    
#     for keyword in crag_keywords:
#         if keyword in question_lower:
#             weight = 3 if keyword in high_value_crag else 1
#             crag_score += weight
    
#     # Additional heuristics
#     # Questions with numbers often indicate data queries
#     if any(char.isdigit() for char in question):
#         sql_score += 2
    
#     # Questions ending with "?" are often factual
#     if question.strip().endswith("?"):
#         sql_score += 1
    
#     # Decision logic with confidence threshold
#     confidence_threshold = 2
    
#     if sql_score > crag_score and sql_score >= confidence_threshold:
#         query_type = "sql"
#     elif crag_score > sql_score and crag_score >= confidence_threshold:
#         query_type = "crag"
#     else:
#         # Improved fallback logic
#         # If contains specific data terms, lean toward SQL
#         data_terms = ["data", "record", "entry", "table", "database", "query"]
#         if any(term in question_lower for term in data_terms):
#             query_type = "sql"
#         else:
#             query_type = "crag"
    
#     return {"query_type": query_type}


# def route_query(state: HybridGraphState) -> Literal["sql_pipeline", "crag_pipeline"]:
#     """
#     Route to either SQL or CRAG pipeline based on classification
#     """
#     return "sql_pipeline" if state["query_type"] == "sql" else "crag_pipeline"


# def create_sql_react_agent(db: SQLDatabase, llm):
#     """
#     Create a React agent for SQL database interaction using LangGraph prebuilt
#     """
#     # Create SQL toolkit
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#     tools = toolkit.get_tools()
    
#     # Enhanced system prompt
#     system_prompt = """
# You are an expert SQL database analyst designed to interact with a SQL database.
# Given an input question, create a syntactically correct {dialect} query to run,
# then look at the results of the query and return the answer.

# IMPORTANT GUIDELINES:
# - Unless the user specifies a specific number of examples, always limit your query to at most {top_k} results
# - You can order the results by a relevant column to return the most interesting examples
# - Never query for all the columns from a specific table, only ask for the relevant columns given the question
# - You MUST double check your query before executing it
# - If you get an error while executing a query, rewrite the query and try again
# - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database
# - To start you should ALWAYS look at the tables in the database to see what you can query
# - Then you should query the schema of the most relevant tables
# - Provide clear, concise answers based on the query results
# - If no results are found, explain that clearly to the user

# WORKFLOW:
# 1. First, examine the available tables in the database
# 2. Identify the most relevant tables for the question
# 3. Query the schema of those tables to understand the structure
# 4. Construct and execute the appropriate SQL query
# 5. Interpret the results and provide a natural language answer
# """.format(
#         dialect=db.dialect,
#         top_k=10,
#     )
    
#     # Create the React agent
#     agent = create_react_agent(
#         llm,
#         tools,
#         prompt=system_prompt,
#     )
    
#     return agent


# def sql_agent_node(state: HybridGraphState, agent) -> HybridGraphState:
#     """
#     Node that uses the React agent to handle SQL queries
#     """
#     question = state["question"]
    
#     # Prepare messages for the agent
#     messages = [HumanMessage(content=question)]
    
#     try:
#         # Invoke the React agent
#         result = agent.invoke({"messages": messages})
        
#         # Extract the response from the agent
#         if result and "messages" in result:
#             last_message = result["messages"][-1]
#             if hasattr(last_message, 'content'):
#                 response = last_message.content
#             else:
#                 response = str(last_message)
#         else:
#             response = str(result)
        
#         return {
#             "generation": response,
#             "agent_outcome": "success"
#         }
    
#     except Exception as e:
#         # If SQL agent fails, we could fall back to CRAG or provide error message
#         error_message = f"Unable to process SQL query: {str(e)}"
#         return {
#             "generation": error_message,
#             "agent_outcome": "failed"
#         }


# def check_sql_success(state: HybridGraphState) -> Literal["end", "fallback_to_crag"]:
#     """
#     Check if SQL agent succeeded or if we need to fall back to CRAG
#     """
#     if state.get("agent_outcome") == "success":
#         return "end"
#     else:
#         return "fallback_to_crag"


# def workflow_compiler(db: SQLDatabase, llm):
#     """
#     Compile the enhanced hybrid workflow with React agent for SQL
#     """
    
#     # Create the SQL React agent
#     sql_agent = create_sql_react_agent(db, llm)
    
#     workflow = StateGraph(HybridGraphState)
    
#     # Add classification node
#     workflow.add_node("classify_query", classify_query_type)
    
#     # SQL Pipeline with React agent
#     workflow.add_node("sql_agent", lambda state: sql_agent_node(state, sql_agent))
    
#     # CRAG Pipeline nodes (your existing nodes)
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("grade_documents", grade_documents)
#     workflow.add_node("generate", generate)
#     workflow.add_node("transform_query", transform_query)
#     workflow.add_node("web_search_node", web_search)
    
#     # Set entry point
#     workflow.set_entry_point("classify_query")
    
#     # Add conditional routing after classification
#     workflow.add_conditional_edges(
#         "classify_query",
#         route_query,
#         {
#             "sql_pipeline": "sql_agent",
#             "crag_pipeline": "retrieve",
#         },
#     )
    
#     # SQL Pipeline edges with fallback capability
#     workflow.add_conditional_edges(
#         "sql_agent",
#         check_sql_success,
#         {
#             "end": END,
#             "fallback_to_crag": "retrieve"
#         }
#     )
    
#     # CRAG Pipeline edges (your existing flow)
#     workflow.add_edge("retrieve", "grade_documents")
#     workflow.add_conditional_edges(
#         "grade_documents",
#         decide_to_generate,
#         {
#             "transform_query": "transform_query",
#             "generate": "generate",
#         },
#     )
#     workflow.add_edge("transform_query", "web_search_node")
#     workflow.add_edge("web_search_node", "generate")
#     workflow.add_edge("generate", END)
    
#     return workflow.compile()


# # Enhanced system creation function
# def create_hybrid_system(database_uri: str, llm, enable_fallback: bool = True):
#     """
#     Create the complete hybrid system with enhanced SQL handling
    
#     Args:
#         database_uri: Database connection string
#         llm: Language model instance
#         enable_fallback: Whether to enable fallback from SQL to CRAG on failure
#     """
#     # Initialize database
#     db = SQLDatabase.from_uri(database_uri)
    
#     # Compile workflow
#     workflow = workflow_compiler(db, llm)
    
#     return workflow, db


# # Enhanced usage example with error handling and logging
# def run_hybrid_query(workflow, question: str, verbose: bool = False):
#     """
#     Run a query through the hybrid system with enhanced error handling
#     """
#     try:
#         initial_state = {
#             "question": question,
#             "generation": "",
#             "web_search": "",
#             "documents": [],
#             "query_type": "",
#             "sql_result": "",
#             "sql_query": "",
#             "table_schemas": "",
#             "messages": [],
#             "agent_outcome": ""
#         }
        
#         result = workflow.invoke(initial_state)
        
#         if verbose:
#             print(f"Query Type: {result.get('query_type', 'Unknown')}")
#             print(f"Question: {question}")
#             print(f"Answer: {result.get('generation', 'No answer generated')}")
        
#         return result
    
#     except Exception as e:
#         error_result = {
#             "question": question,
#             "generation": f"Error processing query: {str(e)}",
#             "query_type": "error"
#         }
#         return error_result

# if __name__ == "__main__":
#     from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
#     import requests
    
#     # Initialize LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash", 
#         google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
#     )
    
#     # Download and setup database
#     url = "https://huggingface.co/datasets/phunghuy159/db_test/resolve/main/eng1.db"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         with open("Chinook.db", "wb") as file:
#             file.write(response.content)
#         print("Database downloaded successfully!")
    
#     db_uri = "sqlite:///Chinook.db"
    
#     # Create enhanced hybrid workflow
#     hybrid_workflow, db = create_hybrid_system(db_uri, llm)
    
#     # Test with different query types
#     test_queries = [
#         "Có bao nhiêu đơn bán hàng được hoạch toán vào ngày 4/4/2024?",  # SQL query
#         # "What are the best practices for customer retention?",  # CRAG query
#         # "How many customers are there in total?",  # SQL query
#         # "List the top 5 best-selling artists",  # SQL query
#     ]
    
#     print("Testing Enhanced Hybrid Workflow:")
#     print("=" * 50)
    
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         result = run_hybrid_query(hybrid_workflow, query, verbose=True)
#         print("-" * 30)


# #     workflow.add_edge("transform_query", "web_search_node")
# #     workflow.add_edge("web_search_node", "generate")
# #     workflow.add_edge("generate", END)
# #     return workflow.compile()





'============================================================================='
# from langgraph.graph import END, StateGraph
# from typing_extensions import TypedDict
# from typing import List, Literal
# from langchain_core.messages import AIMessage
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langgraph.prebuilt import ToolNode
# from langchain.chat_models import init_chat_model

# # Import your existing nodes
# from nodes.transform_node import transform_query
# from nodes.retrieve_node import retrieve
# from nodes.search_node import web_search
# from nodes.grade_node import grade_documents
# from nodes.generate_node import generate
# from nodes.decision_node import decide_to_generate


# class HybridGraphState(TypedDict):
#     question: str
#     generation: str
#     web_search: str
#     documents: List[str]
#     query_type: str  # "sql" or "crag"
#     sql_result: str
#     sql_query: str
#     table_schemas: str


# def classify_query_type(state: HybridGraphState) -> HybridGraphState:
#     """
#     Classify whether the question should be handled by SQL or CRAG
#     """
#     question = state["question"]
    
#     # SQL indicators - customize these based on your database schema
#     sql_keywords = [
#         # Data aggregation keywords
#         "how many", "count", "total", "sum", "average", "maximum", "minimum",
#         "most", "least", "highest", "lowest", "top", "bottom",
        
#         # Time-based queries
#         "in 2023", "in 2024", "last month", "this year", "between",
#         "sales", "revenue", "orders", "customers", "products",
        
#         # Comparison queries  
#         "compare", "versus", "vs", "difference between",
        
#         # Specific data retrieval
#         "list all", "show me", "get all", "find records",
        
#         # Add your specific database entities
#         "employees", "invoices", "artists", "albums", "tracks",'trong cơ sở dữ liệu'
#     ]
    
#     # CRAG indicators - conceptual/analytical questions
#     crag_keywords = [
#         "explain", "what is", "how does", "why", "concept", "definition",
#         "meaning", "understand", "learn about", "tell me about",
#         "describe", "analysis", "insights", "trends", "patterns",
#         "recommendations", "best practices", "methodology"
#     ]
    
#     question_lower = question.lower()
    
#     # Score based on keyword presence
#     sql_score = sum(1 for keyword in sql_keywords if keyword in question_lower)
#     crag_score = sum(1 for keyword in crag_keywords if keyword in question_lower)
    
#     # Decision logic
#     if sql_score > crag_score and sql_score > 0:
#         query_type = "sql"
#     elif crag_score > 0:
#         query_type = "crag"
#     else:
#         # Default fallback - could be made smarter with ML classification
#         if any(char.isdigit() for char in question) or "?" in question:
#             query_type = "sql"
#         else:
#             query_type = "crag"
    
#     return {"query_type": query_type}


# def route_query(state: HybridGraphState) -> Literal["sql_pipeline", "crag_pipeline"]:
#     """
#     Route to either SQL or CRAG pipeline based on classification
#     """
#     return "sql_pipeline" if state["query_type"] == "sql" else "crag_pipeline"


# # SQL Pipeline Nodes
# def get_sql_schema(state: HybridGraphState, db: SQLDatabase) -> HybridGraphState:
#     """
#     Get relevant database schema information
#     """
#     # Get all table names
#     tables = db.get_usable_table_names()
    
#     # Get schema for relevant tables (you can make this smarter)
#     schema_info = []
#     for table in tables:  # Limit to avoid token overflow
#         try:
#             schema = db.get_table_info([table])
#             schema_info.append(schema)
#         except Exception as e:
#             continue
    
#     return {"table_schemas": "\n".join(schema_info)}


# def generate_sql_query(state: HybridGraphState, llm) -> HybridGraphState:
#     """
#     Generate SQL query based on the question and schema
#     """
#     system_prompt = f"""
#     You are a SQL expert. Given the database schema and user question, 
#     generate a syntactically correct SQL query.
    
#     Database Schema:
#     {state.get("table_schemas", "")}
    
#     Rules:
#     - Only use SELECT statements
#     - Limit results to 10 unless specified otherwise
#     - Use proper JOIN syntax when needed
#     - Handle NULL values appropriately
    
#     Question: {state["question"]}
    
#     Return only the SQL query, no explanations.
#     """
    
#     response = llm.invoke(system_prompt)
#     sql_query = response.content.strip()
    
#     # Clean up the query (remove markdown formatting if present)
#     if "```sql" in sql_query:
#         sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
#     elif "```" in sql_query:
#         sql_query = sql_query.split("```")[1].strip()
    
#     return {"sql_query": sql_query}


# def execute_sql_query(state: HybridGraphState, db: SQLDatabase) -> HybridGraphState:
#     """
#     Execute the generated SQL query
#     """
#     try:
#         result = db.run(state["sql_query"])
#         return {"sql_result": str(result)}
#     except Exception as e:
#         # If query fails, could retry with correction or fall back to CRAG
#         return {"sql_result": f"Error executing query: {str(e)}"}


# def generate_sql_response(state: HybridGraphState, llm) -> HybridGraphState:
#     """
#     Generate natural language response from SQL results
#     """
#     prompt = f"""
#     Based on the SQL query results, provide a clear and concise answer to the user's question.
    
#     Question: {state["question"]}
#     SQL Query: {state["sql_query"]}
#     Results: {state["sql_result"]}
    
#     Provide a natural language answer based on these results.
#     """
    
#     response = llm.invoke(prompt)
#     return {"generation": response.content}


# def workflow_compiler(db: SQLDatabase, llm):
#     """
#     Compile the hybrid workflow that can handle both SQL and CRAG queries
#     """
    
#     workflow = StateGraph(HybridGraphState)
    
#     # Add classification node
#     workflow.add_node("classify_query", classify_query_type)
    
#     # SQL Pipeline nodes
#     workflow.add_node("get_sql_schema", lambda state: get_sql_schema(state, db))
#     workflow.add_node("generate_sql_query", lambda state: generate_sql_query(state, llm))
#     workflow.add_node("execute_sql_query", lambda state: execute_sql_query(state, db))
#     workflow.add_node("generate_sql_response", lambda state: generate_sql_response(state, llm))
    
#     # CRAG Pipeline nodes (your existing nodes)
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("grade_documents", grade_documents)
#     workflow.add_node("generate", generate)
#     workflow.add_node("transform_query", transform_query)
#     workflow.add_node("web_search_node", web_search)
    
#     # Set entry point
#     workflow.set_entry_point("classify_query")
    
#     # Add conditional routing after classification
#     workflow.add_conditional_edges(
#         "classify_query",
#         route_query,
#         {
#             "sql_pipeline": "get_sql_schema",
#             "crag_pipeline": "retrieve",
#         },
#     )
    
#     # SQL Pipeline edges
#     workflow.add_edge("get_sql_schema", "generate_sql_query")
#     workflow.add_edge("generate_sql_query", "execute_sql_query")
#     workflow.add_edge("execute_sql_query", "generate_sql_response")
#     workflow.add_edge("generate_sql_response", END)
    
#     # CRAG Pipeline edges (your existing flow)
#     workflow.add_edge("retrieve", "grade_documents")
#     workflow.add_conditional_edges(
#         "grade_documents",
#         decide_to_generate,
#         {
#             "transform_query": "transform_query",
#             "generate": "generate",
#         },
#     )
#     workflow.add_edge("transform_query", "web_search_node")
#     workflow.add_edge("web_search_node", "generate")
#     workflow.add_edge("generate", END)
    
#     return workflow.compile()


# # Usage example
# def create_hybrid_system(database_uri: str, llm):
#     """
#     Create the complete hybrid system
#     """
#     # Initialize database
#     db = SQLDatabase.from_uri(database_uri)
    
#     # Compile workflow
#     workflow = workflow_compiler(db, llm)
    
#     return workflow


# # Example usage:

# # Initialize your LLM and database
# # llm = init_chat_model("openai:gpt-4")
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# import streamlit as st

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
# )
# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
# import requests

# response = requests.get(url)

# if response.status_code == 200:
#     # Open a local file in binary write mode
#     with open("Chinook.db", "wb") as file:
#         # Write the content of the response (the file) to the local file
#         file.write(response.content)
#     print("File downloaded and saved as Chinook.db")
# else:
#     print(f"Failed to download the file. Status code: {response.status_code}")
# db_uri = "sqlite:///Chinook.db"

# # Create hybrid workflow
# hybrid_workflow = create_hybrid_system(db_uri, llm)

# # Test with different query types
# sql_question = "Which sales agent made the most in sales in 2009?"
# crag_question = "What are the best practices for customer retention?"

# # Run SQL query
# # result1 = hybrid_workflow.invoke({"question": sql_question})
# # print(result1)
# # Run CRAG query  
# # result2 = hybrid_workflow.invoke({"question": crag_question})
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict
from typing import List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model

# Import your existing nodes
from nodes.transform_node import transform_query
from nodes.retrieve_node import retrieve
from nodes.search_node import web_search
from nodes.grade_node import grade_documents
from nodes.generate_node import generate
from nodes.decision_node import decide_to_generate


class HybridGraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    query_type: str  # "sql" or "crag"
    sql_result: str
    sql_query: str
    table_schemas: str
    messages: List  # For React agent communication
    agent_outcome: str


def classify_query_type(state: HybridGraphState) -> HybridGraphState:
    """
    Enhanced query classification with better heuristics
    """
    question = state["question"]
    
    # SQL indicators - more comprehensive list
    sql_keywords = [
        # Data aggregation keywords,
        'In my database',
        "how many", "count", "total", "sum", "average", "mean", "maximum", "minimum",
        "most", "least", "highest", "lowest", "top", "bottom", "rank", "sort",
        
        # Time-based queries
        "in 2023", "in 2024", "last month", "this year", "between", "since", "before", "after",
        "during", "quarterly", "monthly", "yearly", "daily",
        
        # Business metrics
        "sales", "revenue", "profit", "orders", "customers", "products", "inventory",
        "transactions", "purchases", "invoices", "employees", "performance",
        
        # Comparison queries  
        "compare", "versus", "vs", "difference between", "better than", "worse than",
        
        # Specific data retrieval
        "list all", "show me", "get all", "find records", "search for", "lookup",
        "who are", "what are", "which", "where are",
        
        # Database-specific entities (customize based on your schema)
        "artists", "albums", "tracks", "genres", "playlists", "media types",
        "invoice", "customer", "employee", "support rep",'nhân viên','sản phẩm','chi nhánh','bán hàng'
    ]
    
    # CRAG indicators - conceptual/analytical questions
    crag_keywords = [
        "explain", "what is", "how does", "why", "concept", "definition",
        "meaning", "understand", "learn about", "tell me about",
        "describe", "analysis", "insights", "trends", "patterns",
        "recommendations", "best practices", "methodology", "strategy",
        "approach", "theory", "principle", "guide", "tutorial"
    ]
    
    question_lower = question.lower()
    
    # Enhanced scoring with weighted keywords
    sql_score = 0
    crag_score = 0
    
    # Weight important SQL indicators higher
    # high_value_sql = ["how many", "count", "total", "sum", "average", "list all", "top", "most",'']
    high_value_sql = ['cơ sở dữ liệu', 'trong cơ sở dữ liệu', 'nhân viên']
    high_value_crag = ["explain", "what is", "best practices", "methodology"]
    
    for keyword in sql_keywords:
        if keyword in question_lower:
            weight = 5 if keyword in high_value_sql else 1
            sql_score += weight
    
    for keyword in crag_keywords:
        if keyword in question_lower:
            weight = 3 if keyword in high_value_crag else 1
            crag_score += weight
    
    # Additional heuristics
    # Questions with numbers often indicate data queries
    if any(char.isdigit() for char in question):
        sql_score += 2
    
    # Questions ending with "?" are often factual
    if question.strip().endswith("?"):
        sql_score += 1
    
    # Decision logic with confidence threshold
    confidence_threshold = 2
    
    if sql_score > crag_score and sql_score >= confidence_threshold:
        query_type = "sql"
    elif crag_score > sql_score and crag_score >= confidence_threshold:
        query_type = "crag"
    else:
        # Improved fallback logic
        # If contains specific data terms, lean toward SQL
        data_terms = ["data", "record", "entry", "table", "database", "query"]
        if any(term in question_lower for term in data_terms):
            query_type = "sql"
        else:
            query_type = "crag"
    
    return {"query_type": query_type}


def route_query(state: HybridGraphState) -> Literal["sql_pipeline", "crag_pipeline"]:
    """
    Route to either SQL or CRAG pipeline based on classification
    """
    return "sql_pipeline" if state["query_type"] == "sql" else "crag_pipeline"


def create_sql_react_agent(db: SQLDatabase, llm):
    """
    Create a React agent for SQL database interaction using LangGraph prebuilt
    """
    # Create SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    # Enhanced system prompt
    system_prompt = """
You are an expert SQL database analyst designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer.

IMPORTANT GUIDELINES:
- Unless the user specifies a specific number of examples, always limit your query to at most {top_k} results
- You can order the results by a relevant column to return the most interesting examples
- Never query for all the columns from a specific table, only ask for the relevant columns given the question
- You MUST double check your query before executing it
- If you get an error while executing a query, rewrite the query and try again
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database
- To start you should ALWAYS look at the tables in the database to see what you can query
- Then you should query the schema of the most relevant tables
- Provide clear, concise answers based on the query results
- If no results are found, explain that clearly to the user

WORKFLOW:
1. First, examine the available tables in the database
2. Identify the most relevant tables for the question
3. Query the schema of those tables to understand the structure
4. Construct and execute the appropriate SQL query
5. Interpret the results and provide a natural language answer
""".format(
        dialect=db.dialect,
        top_k=10,
    )
    
    # Create the React agent
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )
    
    return agent


def sql_agent_node(state: HybridGraphState, agent) -> HybridGraphState:
    """
    Node that uses the React agent to handle SQL queries
    """
    question = state["question"]
    
    # Prepare messages for the agent
    messages = [HumanMessage(content=question)]
    
    try:
        # Invoke the React agent
        result = agent.invoke({"messages": messages})
        
        # Extract the response from the agent
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                response = last_message.content
            else:
                response = str(last_message)
        else:
            response = str(result)
        
        return {
            "generation": response,
            "agent_outcome": "success"
        }
    
    except Exception as e:
        # If SQL agent fails, we could fall back to CRAG or provide error message
        error_message = f"Unable to process SQL query: {str(e)}"
        return {
            "generation": error_message,
            "agent_outcome": "failed"
        }


def check_sql_success(state: HybridGraphState) -> Literal["end", "fallback_to_crag"]:
    """
    Check if SQL agent succeeded or if we need to fall back to CRAG
    """
    if state.get("agent_outcome") == "success":
        return "end"
    else:
        return "fallback_to_crag"


def workflow_compiler(db: SQLDatabase, llm):
    """
    Compile the enhanced hybrid workflow with React agent for SQL
    """
    
    # Create the SQL React agent
    sql_agent = create_sql_react_agent(db, llm)
    
    workflow = StateGraph(HybridGraphState)
    
    # Add classification node
    workflow.add_node("classify_query", classify_query_type)
    
    # SQL Pipeline with React agent
    workflow.add_node("sql_agent", lambda state: sql_agent_node(state, sql_agent))
    
    # CRAG Pipeline nodes (your existing nodes)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Add conditional routing after classification
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "sql_pipeline": "sql_agent",
            "crag_pipeline": "retrieve",
        },
    )
    
    # SQL Pipeline edges with fallback capability
    workflow.add_conditional_edges(
        "sql_agent",
        check_sql_success,
        {
            "end": END,
            "fallback_to_crag": "retrieve"
        }
    )
    
    # CRAG Pipeline edges (your existing flow)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Enhanced system creation function
def create_hybrid_system(database_uri: str, llm, enable_fallback: bool = True):
    """
    Create the complete hybrid system with enhanced SQL handling
    
    Args:
        database_uri: Database connection string
        llm: Language model instance
        enable_fallback: Whether to enable fallback from SQL to CRAG on failure
    """
    # Initialize database
    db = SQLDatabase.from_uri(database_uri)
    
    # Compile workflow
    workflow = workflow_compiler(db, llm)
    
    return workflow, db


# Enhanced usage example with error handling and logging
def run_hybrid_query(workflow, question: str, verbose: bool = False):
    """
    Run a query through the hybrid system with enhanced error handling
    """
    try:
        initial_state = {
            "question": question,
            "generation": "",
            "web_search": "",
            "documents": [],
            "query_type": "",
            "sql_result": "",
            "sql_query": "",
            "table_schemas": "",
            "messages": [],
            "agent_outcome": ""
        }
        
        result = workflow.invoke(initial_state)
        
        if verbose:
            print(f"Query Type: {result.get('query_type', 'Unknown')}")
            print(f"Question: {question}")
            print(f"Answer: {result.get('generation', 'No answer generated')}")
        
        return result
    
    except Exception as e:
        error_result = {
            "question": question,
            "generation": f"Error processing query: {str(e)}",
            "query_type": "error"
        }
        return error_result

if __name__ == "__main__":
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    import requests
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
    )
    
    # Download and setup database
    url = "https://huggingface.co/datasets/phunghuy159/db_test/resolve/main/eng1.db"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("Chinook.db", "wb") as file:
            file.write(response.content)
        print("Database downloaded successfully!")
    
    db_uri = "sqlite:///Chinook.db"
    
    # Create enhanced hybrid workflow
    hybrid_workflow, db = create_hybrid_system(db_uri, llm)
    
    # Test with different query types
    test_queries = [
        "Có bao nhiêu đơn bán hàng được hoạch toán vào ngày 4/4/2024?",  # SQL query
        # "What are the best practices for customer retention?",  # CRAG query
        # "How many customers are there in total?",  # SQL query
        # "List the top 5 best-selling artists",  # SQL query
    ]
    
    print("Testing Enhanced Hybrid Workflow:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = run_hybrid_query(hybrid_workflow, query, verbose=True)
        print("-" * 30)

