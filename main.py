from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import tempfile
import os
from git import Repo
from openai import OpenAI
from pathlib import Path
from pinecone import Pinecone
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings
from github import  Repository
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import uuid

def add_documents_to_pinecone(vector_store, documents, namespace, batch_size=50, max_workers=10):
    print(f"Total documents to upload: {len(documents)}")

    def upload_batch(batch, batch_id):
        try:
            vector_store.add_documents(batch, namespace=namespace)
            print(f"[✓] Uploaded batch {batch_id} to namespace {namespace}")
        except Exception as e:
            print(f"[✗] Failed batch {batch_id}: {e}")

    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(upload_batch, batch, idx + 1)
            for idx, batch in enumerate(batches)
        ]
        concurrent.futures.wait(futures)

    print(f"✅ All document batches uploaded to namespace: {namespace}")

# pip install -q langchain gitpython sentence-transformers
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    if os.path.exists(os.path.join(tempfile.gettempdir(), repo_name)):
        return os.path.join(tempfile.gettempdir(), repo_name)  # Check if the repository already exists
    
    # Check if the repository already exists
    repo_path = Path(tempfile.gettempdir()) / repo_name  # Create a temporary path
    Repo.clone_from(repo_url, str(repo_path))
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

SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

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

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
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
        # Create index if it doesn't exist
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=768,  # Dimension for all-mpnet-base-v2 model
            metric="cosine",
            spec= ServerlessSpec(
                cloud="aws",
                region=st.secrets["PINECONE_ENV"]
            )
        )
        print(f"Index created: {index_name}")
    else:
        print(f"Index already exists: {index_name}")
    # Get the index
    index = pc.Index(index_name)
    return index

# def add_documents_to_pinecone(vector_store, documents, batch_size=50):
#     print(f"Total documents to upload: {len(documents)}")
#     for i in range(0, len(documents), batch_size):
#         batch = documents[i:i+batch_size]
#         vector_store.add_documents(batch, namespace="github-repo-index")
#         print(f"Uploaded batch {i//batch_size + 1} of {((len(documents) - 1)//batch_size) + 1}")
#     print("All documents uploaded.")


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

    return llm_response.choices[0].message.content
    
def create_vector_store(index_name, index):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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
        
    namespace = st.session_state.namespace
    
    # Display current namespace (optional, for debugging)
    st.sidebar.write(f"Current namespace: `{namespace}`")
    
    repo_url = st.text_input("Enter GitHub Repository URL:", value="https://github.com/user/repo")
    index_name = "github-repo-index"
    index = initialize_pinecone_index(index_name)
    vector_store = create_vector_store(index_name, index)

    if st.button("Clone and Index Repository"):
        if repo_url:
            st.session_state.repo_path = clone_repository(repo_url)  # Store in session state
            documents = get_all_file_contents(st.session_state.repo_path)
            documents = split_documents(documents)
            add_documents_to_pinecone(vector_store, documents, namespace)
            st.success("Documents added to Pinecone index successfully!")
        else:
            st.error("Please enter a valid GitHub repository URL.")

    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API"])
            response = apply_rag(query, vector_store, client, namespace)
            st.write(response)
        else:
            st.error("Please enter a query to search.")
            
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
