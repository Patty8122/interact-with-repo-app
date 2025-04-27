import os
import tempfile
import uuid
import time
import json
import streamlit as st
# from tqdm.contrib.concurrent import thread_map 
from stqdm import stqdm

import concurrent.futures
from pathlib import Path
from git import Repo
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}


# file based metrics
# 1. Total number of tokens in the repository
# 2. Time taken to index the repository: Indexing Time per 1000 tokens
# 3. Time taken per query: Query Latency
# 4. Number of queries made
# 5. Average query latency
# 6. Total query latency
overall_metrics_file = "overall_metrics.json"
if not os.path.exists(overall_metrics_file):
    with open(overall_metrics_file, 'w') as f:
        json.dump({
            "total_tokens": 0,
            "indexing_time_per_1000_tokens": 0,
            "current_query_latency": 0,
            "number_of_queries": 0,
            "total_query_latency": 0
        }, f, indent=4)

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 1000 tokens (approximately 4000 characters)
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def add_documents_to_pinecone_sequentially(vector_store, documents, batch_size=50):
    print(f"Total documents to upload: {len(documents)}")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vector_store.add_documents(batch, namespace="github-repo-index")
        print(f"Uploaded batch {i//batch_size + 1} of {((len(documents) - 1)//batch_size) + 1}")
    print("All documents uploaded.")
    
    
def upload_batch(vector_store, namespace, batch, batch_id):
    try:
        vector_store.add_documents(batch, namespace=namespace)
        print(f"[✓] Uploaded batch {batch_id} to namespace {namespace}")
    except Exception as e:
        print(f"[✗] Failed batch {batch_id}: {e}")

def add_documents_to_pinecone(vector_store, documents, namespace, batch_size=50, max_workers=10):
    """
    Uploads documents to Pinecone in batches using multithreading.
    Returns: time taken to upload all documents.
    Args:
        vector_store: The Pinecone vector store object.
        documents: A list of Document objects to upload.
        namespace: The namespace to upload the documents to.
        batch_size: The number of documents to upload in each batch.
        max_workers: The maximum number of threads to use for uploading.
    """
    print(f"Total documents to upload: {len(documents)}")


    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # futures = [
        #     executor.submit(upload_batch, vector_store, namespace, batch, idx + 1)
        #     for idx, batch in enumerate(batches)
        # ]
        # futures = thread_map(
        #     upload_batch,
        #     [vector_store] * len(batches),
        #     [namespace] * len(batches),
        #     batches,
        #     range(1, len(batches) + 1),
        #     max_workers=max_workers,
        #     chunksize=1
        # )
        futures = [
            executor.submit(upload_batch, vector_store, namespace, batch, idx + 1)
            for idx, batch in enumerate(batches)
        ]
        for future in stqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading batches"):
            future.result()  # raises exception if any batch upload failed
    
    print(f"✅ All document batches uploaded to namespace: {namespace}")
    return time.time() - start_time  # Return the total time taken to upload all documents

def clone_repository(repo_url, namespace):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    repo_name = namespace
    if os.path.exists(os.path.join(tempfile.gettempdir(), repo_name)):
        if os.path.exists(os.path.join(tempfile.gettempdir(), repo_name, "_metrics.json")):
            print(f"Repository {repo_name} already exists in temporary directory.")
            # Return the path to the existing repository
        else:
            with open(os.path.join(tempfile.gettempdir(), repo_name, "_metrics.json"), 'w') as f:
                json.dump({}, f)

        return os.path.join(tempfile.gettempdir(), repo_name)  # Check if the repository already exists
    
    # Check if the repository already exists
    repo_path = Path(tempfile.gettempdir()) / repo_name  # Create a temporary path
    Repo.clone_from(repo_url, str(repo_path))
    # Create a metrics file in the same directory
    metrics_file_path = repo_path / f"{repo_name}_metrics.json"
    if not metrics_file_path.exists():
        with open(metrics_file_path, 'w') as f:
            json.dump({}, f)  # Initialize an empty metrics file
    return str(repo_path)

def get_files_from_directory(directory):
    """Recursively retrieves all files from a directory.

    Args:
        directory: The path to the directory.

    Returns:
        A list of file paths.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def get_file_content(file_path):
    """Reads the content of a file.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_all_file_contents(repo_path):
    """Retrieves the content of all supported files in a repository.

    Args:
        repo_path: The path to the cloned repository.

    Returns:
        A list of Document objects containing file content and metadata.
    """
    all_files = get_files_from_directory(repo_path)
    documents = []
    for file_path in all_files:
        if os.path.splitext(file_path)[1] in SUPPORTED_EXTENSIONS and \
                not any(ignored_dir in file_path for ignored_dir in IGNORED_DIRS):
            content = get_file_content(file_path)
            documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

def get_huggingface_embeddings(text, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text) 
    
    
def initialize_pinecone_index(index_name):
    """Creates a Pinecone index.

    Args:
        index_name: The name of the index.

    Returns:
        The Pinecone index object.
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    
    # Check if index exists    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=768,  # Dimension for all-mpnet-base-v2 model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=st.secrets["PINECONE_ENV"]
            )
        )
        print(f"Index created: {index_name}")

    # Get the index
    index = pc.Index(index_name)
    return index


def search_similar_documents(vector_store, query, namespace, top_k=5):
    """Searches for similar documents in the Pinecone index.

    Args:
        vector_store: The Pinecone vector store object.
        query: The query string.
        namespace: The namespace to search in.
        top_k: The number of top similar documents to retrieve.

    Returns:
        A list of similar Document objects.
    """
    results = vector_store.similarity_search(query, k=top_k, namespace=namespace)
    return results

def apply_rag(query, vector_store, client, namespace):
    """Apply RAG using Pinecone and LLM."""
    results = vector_store.similarity_search(query, k=5, namespace=namespace)
    print(f"Searching in namespace: {namespace}")
    
    # Method 1: Manually extract contexts from results
    contexts = [doc.page_content for doc in results]
    
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    
    system_prompt = """
    You are a Senior Software Engineer, specializing in TypeScript.
    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """
    
    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    
    # Method 2: Use LangChain's RAG chain
    # retriever = get_vectorstore().as_retriever()
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     memory=memory # System prompt is added to the memory
    # )

    return llm_response.choices[0].message.content

    
def create_vector_store(index_name, index):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return PineconeVectorStore(index=index, embedding=embeddings)

def delete_namespace(index, namespace="github-repo-index"):
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Namespace '{namespace}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting namespace '{namespace}': {e}")

def main():
    st.title("GitHub Repository Search")
    
    # Initialize namespace and repo_path in session state
    if "namespace" not in st.session_state:
        st.session_state.namespace = f"github-repo-{uuid.uuid4().hex[:8]}"
    if "repo_path" not in st.session_state:
        st.session_state.repo_path = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_tokens": 0,
            "indexing_time_per_1000_tokens": 0,
            "current_query_latency": 0,
            "number_of_queries": 0,
            "total_query_latency": 0
        }
    namespace = st.session_state.namespace
    
    # Display current namespace (optional, for debugging)
    st.sidebar.write(f"Current namespace: `{namespace}`")

    repo_url = st.text_input("Enter GitHub Repository URL:", value="https://github.com/sbarry25/SuperMarioBros---RL")
    index_name = "github-repo-index"
    index = initialize_pinecone_index(index_name)
    vector_store = create_vector_store(index_name, index)

    if st.button("Clone and Index Repository"):
        if repo_url:
            st.session_state.repo_path = clone_repository(repo_url, namespace)  # Store in session state
            documents = get_all_file_contents(st.session_state.repo_path)
            documents = split_documents(documents)
            st.session_state.metrics["total_tokens"] = sum(len(doc.page_content.split()) for doc in documents)
            st.sidebar.write(f"Total tokens in repository: {st.session_state.metrics['total_tokens']}")
            duration = add_documents_to_pinecone(vector_store, documents, namespace)
            st.success(f"Time taken to index the repository: {duration:.2f} seconds")
            st.session_state.metrics["indexing_time_per_1000_tokens"] = duration / (st.session_state.metrics["total_tokens"] / 1000)
            st.success("Documents added to Pinecone index successfully!")
        else:
            st.error("Please enter a valid GitHub repository URL.")
    
    # Initialize messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display existing messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API"])
    
    # Handle new messages
    if prompt := st.chat_input("Ask me anything about the codebase"):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        start_time = time.time()  # Start timer for query latency
        response = apply_rag(prompt, vector_store, client, namespace)
        
        duration = time.time() - start_time  # Calculate query latency
        st.session_state.metrics["current_query_latency"] = duration
        st.session_state.metrics["number_of_queries"] = st.session_state.get("number_of_queries", 0) + 1
        st.session_state.metrics["total_query_latency"] += duration
        if st.session_state.metrics["number_of_queries"] > 0:
            st.session_state.metrics["average_query_latency"] = st.session_state.metrics["total_query_latency"] / st.session_state.metrics["number_of_queries"]
        else:
            st.session_state.metrics["average_query_latency"] = 0
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Rerun to display new messages
        st.rerun()
        
    # if st.button("Report Metrics"):
    metrics_file_path = Path(tempfile.gettempdir()) / f"{st.session_state.namespace}_metrics.json"
    # Save metrics to a JSON file
    with open(metrics_file_path, 'w') as f:
        json.dump(st.session_state.metrics, f, indent=4)
    st.success(f"Metrics reported to {metrics_file_path}")   
    # update overall metrics file
    with open(overall_metrics_file, 'r+') as f:
        try:
            overall_metrics = json.load(f)
        except json.JSONDecodeError:
            overall_metrics = {
                "total_tokens": 0,
                "indexing_time_per_1000_tokens": 0,
                "current_query_latency": 0,
                "number_of_queries": 0,
                "total_query_latency": 0
            }
        overall_metrics["total_tokens"] += st.session_state.metrics["total_tokens"]
        overall_metrics["indexing_time_per_1000_tokens"] = st.session_state.metrics["indexing_time_per_1000_tokens"]
        overall_metrics["current_query_latency"] = st.session_state.metrics["current_query_latency"]
        overall_metrics["number_of_queries"] += st.session_state.metrics["number_of_queries"]
        overall_metrics["total_query_latency"] += st.session_state.metrics["total_query_latency"]
        overall_metrics["average_query_latency"] = overall_metrics["total_query_latency"] / overall_metrics["number_of_queries"] if overall_metrics["number_of_queries"] > 0 else 0
        f.seek(0)  # Move to the beginning of the file
        json.dump(overall_metrics, f, indent=4)
        f.truncate()  # Ensure the file is truncated after writing

    # Display overall metrics
    st.sidebar.subheader("Overall Metrics")
    with open(overall_metrics_file, 'r') as f:
        overall_metrics = json.load(f)
    for key, value in overall_metrics.items():
        st.sidebar.write(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    if st.button("Delete Namespace"):
        delete_namespace(index, namespace)
        # Generate new namespace after deletion
        st.session_state.namespace = f"github-repo-{uuid.uuid4().hex[:8]}"
        st.success(f"Namespace '{namespace}' deleted successfully!")
        
        # Clean up temporary files if they exist
        if st.session_state.repo_path and os.path.exists(st.session_state.repo_path):
            for root, dirs, files in os.walk(st.session_state.repo_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(st.session_state.repo_path)
            print(f"Temporary directory {st.session_state.repo_path} cleaned up.")
            st.session_state.repo_path = None  # Reset the repo_path
    
        


if __name__ == "__main__":
    main()
