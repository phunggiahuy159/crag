
# import sys
# from models.LLM import llm
# from tools.index_tool import indexer
# from graph import workflow_compiler
# print('djtttttt')
# def generate_llm_response(input_text):
#     ans = ""
#     for token in llm.stream(input_text):
#         ans += token.content
#     print("\nLLM Response:\n")
#     print(ans)

# def generate_rag_response(app, input_text):
#     ans = ""
#     input_dict = {"question": str(input_text)}
#     response = app.invoke(input_dict)

#     for token in response["generation"]:
#         ans += token
#     print("\nRAG Response:\n")
#     print(ans)

#     print("\nSources:")
#     for j, doc in enumerate(response["documents"]):
#         s = str(doc.page_content).replace("\n", " ")
#         doc_snippet = s if len(s) <= 100 else f"{s[:45]}...{s[-45:]}"
#         print(f"{j+1}. Document: ({doc_snippet})")
#         print(f"   Source: {doc.metadata['source']}")
#         if "page" in doc.metadata:
#             print(f"   Page: {int(doc.metadata['page']) + 1}")

# def main():
#     # if len(sys.argv) != 2:
#     #     print("Usage: python run_rag.py path_to_pdf")
#     #     sys.exit(1)

#     # file_path = sys.argv[1]
    
#     # try:
#     #     # print(f"Indexing document: {file_path}")
#     #     # indexer(file_path)
#     app = workflow_compiler()
#     #     print("Indexing completed.\n")
#     # except Exception as e:
#     #     print(f"Error indexing file: {e}")
#     #     sys.exit(1)

#     # Accept user input
#     while True:
#         try:
#             query = input("\nEnter your query (or 'exit' to quit): ")
#             if query.lower() in ['exit', 'quit']:
#                 break
#             generate_rag_response(app, query)
#         except KeyboardInterrupt:
#             print("\nExiting...")
#             break

# if __name__ == "__main__":
#     main()

# import sys
# from datasets import load_dataset  # to load Hugging Face dataset
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.faiss import FAISS
# from langchain.schema import Document  # Import Document class
# from models.EM import embedding

# # Function to index the dataset
# def index_huggingface_dataset(dataset_name, split="train"):
#     try:
#         # Load the dataset from Hugging Face
#         dataset = load_dataset(dataset_name)
#         print(len(dataset['train']))
#         docs = dataset[split][0:1288679]
#         print(len(docs))
#         # print(docs)# Access the 'train' split (or another split)
#         print(f"Loaded dataset: {dataset_name}, Split: {split}")
        
#         # Split the text in the dataset
#         splits = []
#         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             chunk_size=1024, chunk_overlap=128
#         )
#         print('ok')
        
#         # Iterate over each document in the dataset and split the text
#         for item in docs:
#             text = item
#             # print('dkm')  # Assuming the text is in the "text" field
#             split_docs = text_splitter.split_text(text)  # Split the document into chunks
            
#             # Convert each chunk into a Document object
#             for doc in split_docs:
#                 # You can add metadata if needed, e.g., doc_id or document index
#                 document = Document(page_content=doc)
#                 splits.append(document)  # Add the Document to the splits list
        
#         print(f"Total number of document chunks: {len(splits)}")
        
#         # Create embeddings for the chunks
#         print("Creating embeddings for chunks...")
#         vectorstore = FAISS.from_documents(splits, embedding)
        
#         # Save the vector store to local storage
#         vectorstore.save_local("wiki_")
#         print("Indexing completed and saved locally.")
        
#     except Exception as e:
#         print(f"Error indexing dataset: {e}")
#         sys.exit(1)

# # If you're running from a command line
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python index_huggingface.py <dataset_name>")
#         sys.exit(1)

#     dataset_name = sys.argv[1]  # Pass dataset name as a command-line argument
#     index_huggingface_dataset(dataset_name)  # Index the dataset

import sys
from datasets import load_dataset  # to load Hugging Face dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document  # Import Document class
from models.EM import embedding

# Function to index the dataset
def index_huggingface_dataset(dataset_name, split="train"):
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        docs = dataset[split]  # Access the 'train' split (or another split)

        print(f"Loaded dataset: {dataset_name}, Split: {split}, Number of documents: {len(docs)}")
        print(docs[0])
        docs = docs[1050000:]
        # Convert to a list and slice (if needed)
  # Slicing the list to get only the first N documents
        # print(f"Total number of documents after slicing: {len(docs)}")
        
        # Split the text in the dataset
        splits = []
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024, chunk_overlap=128
        )
        # print(docs)
        
        # Iterate over each document in the dataset and split the text
# Iterate over each document in the dataset and split the text
        for idx, item in enumerate(docs['text']):
            # print(item)
            if idx%5000==0:
              print(idx)
            # If the document is a string (not a dictionary), use it directly
            if isinstance(item, str):
                text = item
                metadata = {'source': f"Document-{idx}"}
            else:
                text = item.get("text", "")  # Modify this if the text is under a different key
                metadata = {'source': f"Document-{idx}"}

            if text:  # Check if there's text to split
                split_docs = text_splitter.split_text(text)  # Split the document into chunks
                
                # Convert each chunk into a Document object
                for doc in split_docs:
                    document = Document(page_content=doc, metadata=metadata)  # Add metadata
                    splits.append(document)  # Add the Document to the splits list

        print(f"Total number of document chunks: {len(splits)}")
        
        # Create embeddings for the chunks
        print("Creating embeddings for chunks...")
        vectorstore = FAISS.from_documents(splits, embedding)
        
        # Save the vector store to local storage
        vectorstore.save_local("wiki_600")  # Change this to a unique path if needed
        print("Indexing completed and saved locally.")
        
    except Exception as e:
        print(f"Error indexing dataset: {e}")
        sys.exit(1)

# If you're running from a command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python index_huggingface.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]  # Pass dataset name as a command-line argument
    index_huggingface_dataset(dataset_name)  # Index the dataset



# import sys
# from models.LLM import llm
# from tools.index_tool import indexer
# from graph import workflow_compiler
# from langchain_community.utilities import SQLDatabase

# print('djtttttt')
# def generate_llm_response(input_text):
#     ans = ""
#     for token in llm.stream(input_text):
#         ans += token.content
#     print("\nLLM Response:\n")
#     print(ans)

# def generate_rag_response(app, input_text):
#     ans = ""
#     input_dict = {"question": str(input_text)}
#     response = app.invoke(input_dict)

#     for token in response["generation"]:
#         ans += token
#     print("\nRAG Response:\n")
#     print(ans)

#     print("\nSources:")
#     for j, doc in enumerate(response["documents"]):
#         s = str(doc.page_content).replace("\n", " ")
#         doc_snippet = s if len(s) <= 100 else f"{s[:45]}...{s[-45:]}"
#         print(f"{j+1}. Document: ({doc_snippet})")
#         print(f"   Source: {doc.metadata['source']}")
#         if "page" in doc.metadata:
#             print(f"   Page: {int(doc.metadata['page']) + 1}")

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python run_rag.py path_to_pdf")
#         sys.exit(1)

#     file_path = sys.argv[1]
    
#     try:
#         print(f"Indexing document: {file_path}")
#         indexer(file_path)
#         from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

#         llm = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash", google_api_key='AIzaSyCyXr2KjwW58Vm0bewJ_sGEau8C1WS_QNQ'
#         )
#         # url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
#         url = 'https://huggingface.co/datasets/phunghuy159/db_test/resolve/main/eng1.db'
#         import requests

#         response = requests.get(url)

#         if response.status_code == 200:
#             # Open a local file in binary write mode
#             with open("Chinook.db", "wb") as file:
#                 print(response.content)
#                 # Write the content of the response (the file) to the local file
#                 file.write(response.content)
#             print("File downloaded and saved as Chinook.db")
#         else:
#             print(f"Failed to download the file. Status code: {response.status_code}")
#         db_uri = "sqlite:///Chinook.db"
#         # db_uri = "sqlite:///eng1.db"

#         db = SQLDatabase.from_uri(db_uri)
#         print(f"Available tables: {db.get_usable_table_names()}")

#         # db_uri = '/content/crag/Chinook.db'
#         app = workflow_compiler(db,llm)
#         print("Indexing completed.\n")
#     except Exception as e:
#         print(f"Error indexing file: {e}")
#         sys.exit(1)

#     # Accept user input
#     while True:
#         try:
#             query = input("\nEnter your query (or 'exit' to quit): ")
#             if query.lower() in ['exit', 'quit']:
#                 break
#             generate_rag_response(app, query)
#         except KeyboardInterrupt:
#             print("\nExiting...")
#             break

# if __name__ == "__main__":
#     main()
