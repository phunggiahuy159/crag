import gradio as gr
import requests
from models.LLM import llm
from graph import workflow_compiler
from langchain_community.utilities import SQLDatabase
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class RAGSystem:
    def __init__(self):
        self.app = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the RAG system with database and LLM"""
        try:
            # Initialize LLM
            llm_instance = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
            )
            
            # Download and setup database
            url = 'https://huggingface.co/datasets/phunghuy159/db_test/resolve/main/eng1.db'
            response = requests.get(url)
            
            if response.status_code == 200:
                with open("Chinook.db", "wb") as file:
                    file.write(response.content)
                print("Database downloaded successfully")
            else:
                raise Exception(f"Failed to download database. Status code: {response.status_code}")
            
            # Setup database connection
            db_uri = "sqlite:///Chinook.db"
            db = SQLDatabase.from_uri(db_uri)
            print(f"Available tables: {db.get_usable_table_names()}")
            
            # Initialize workflow
            self.app = workflow_compiler(db, llm_instance)
            return "✅ RAG System initialized successfully!"
            
        except Exception as e:
            error_msg = f"❌ Error initializing system: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_response(self, query):
        """Generate RAG response for the given query"""
        if not self.app:
            return "❌ System not initialized. Please restart the application.", ""
        
        if not query.strip():
            return "Please enter a valid query.", ""
        
        try:
            # Prepare input
            input_dict = {"question": str(query)}
            response = self.app.invoke(input_dict)
            
            # Extract response text
            answer = ""
            for token in response["generation"]:
                answer += token
            
            # Format sources
            sources_info = ""
            if "documents" in response and response["documents"]:
                sources_info = "**Sources:**\n"
                for j, doc in enumerate(response["documents"]):
                    s = str(doc.page_content).replace("\n", " ")
                    doc_snippet = s if len(s) <= 100 else f"{s[:45]}...{s[-45:]}"
                    sources_info += f"{j+1}. **Document:** {doc_snippet}\n"
                    sources_info += f"   **Source:** {doc.metadata['source']}\n"
                    if "page" in doc.metadata:
                        sources_info += f"   **Page:** {int(doc.metadata['page']) + 1}\n"
                    sources_info += "\n"
            else:
                sources_info = "No sources available for this response."
            
            return answer, sources_info
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            return error_msg, ""

# Initialize RAG system
rag_system = RAGSystem()

def chat_interface(message, history):
    """Chat interface function for Gradio"""
    answer, sources = rag_system.generate_response(message)
    
    # Combine answer and sources for display
    full_response = answer
    if sources:
        full_response += f"\n\n---\n\n{sources}"
    
    return full_response

def clear_chat():
    """Clear the chat history"""
    return []

# Create Gradio interface
with gr.Blocks(
    title="RAG Question Answering System",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
    }
    .chat-container {
        height: 600px;
    }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🤖 RAG Question Answering System
        
        Ask questions about your database and get intelligent responses powered by RAG (Retrieval-Augmented Generation).
        The system uses a SQLite database and Gemini 2.0 Flash for generating contextual answers.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter your question here...",
                    label="Your Question",
                    lines=2,
                    max_lines=5,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### 💡 Tips:
                - Ask specific questions about the database
                - You can ask about relationships between data
                - Try questions like:
                  - "What tables are available?"
                  - "Show me customer information"
                  - "What are the top selling products?"
                
                ### 🔧 System Status:
                - Database: ✅ Connected
                - LLM: ✅ Gemini 2.0 Flash
                - RAG: ✅ Active
                """
            )
    
    # Event handlers
    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""
        
        # Get response from RAG system
        bot_response = chat_interface(message, chat_history)
        
        # Update chat history
        chat_history.append((message, bot_response))
        return chat_history, ""
    
    # Button and enter key events
    submit_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, msg]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )
