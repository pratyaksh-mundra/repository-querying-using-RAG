# Import necessary libraries and modules
import os
from langchain.document_loaders import PyPDFLoader  # Loader for PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into chunks for processing
from langchain.embeddings import BedrockEmbeddings  # Embeddings for vector representation of text
from langchain.vectorstores import FAISS  # FAISS is a library for efficient similarity search
from langchain.indexes import VectorstoreIndexCreator  # Creates a vector index for fast document retrieval
from langchain.llms.bedrock import Bedrock  # Model interface for AWS Bedrock LLMs
from pathlib import Path  # Path library for file and directory handling
from langchain.chains import ConversationChain, ConversationalRetrievalChain  # Chains for conversation handling
from langchain.memory import ConversationSummaryBufferMemory  # Memory for summarizing conversation history
from langchain_aws import BedrockLLM  # AWS LLM integration
from langchain_community.document_loaders import DirectoryLoader  # Loader to process files in a directory
from langchain.schema import Document  # Schema for managing documents
from langchain.text_splitter import CharacterTextSplitter  # Splits text into chunks for small passages
from langchain.embeddings import BedrockEmbeddings  # For generating embeddings of text
from langchain.vectorstores import FAISS  # Fast similarity search
from langchain.indexes import VectorstoreIndexCreator  # For creating a vector index from documents
from pathlib import Path  # Handles file paths
import nltk  # Natural language processing library
from langchain.prompts import PromptTemplate  # Template for creating prompts for LLMs

# Download necessary NLTK data (Punkt tokenizer for splitting sentences)
nltk.download('punkt')

# Function to load files from a GitHub repository directory, filtering by file type.
def load_github_repo(repo_path, file_types=['.py', '.md', '.sh']):
    """
    Load all files from a given GitHub repository directory.
    
    :param repo_path: Path to the repository.
    :param file_types: List of file types to include (default is .py, .md, and .sh).
    :return: A list of loaded documents from the repository.
    """
    try:
        # Load markdown files from the repository
        loader = DirectoryLoader(repo_path, glob='**/*.md')
        all_documents = loader.load()

        # Filter documents based on file extensions (.py, .md, excluding .sh)
        filtered_documents = [doc for doc in all_documents if any(doc.metadata['source'].endswith(ext) for ext in file_types if ext != '.sh')]

        # Load and process shell script files (.sh)
        for file_path in Path(repo_path).rglob('*.sh'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    document = Document(page_content=content, metadata={'source': str(file_path)})
                    filtered_documents.append(document)
            except Exception as e:
                print(f"Error loading .sh file {file_path}: {e}")

        return filtered_documents

    except Exception as e:
        raise Exception(f"Error loading GitHub repository: {e}")

# Function to split code documents into smaller chunks for better processing with LLMs.
def split_code_documents(documents, chunk_size=200, chunk_overlap=20):
    """
    Split code documents into smaller chunks using CharacterTextSplitter.
    
    :param documents: List of documents to split.
    :param chunk_size: Size of each chunk in characters.
    :param chunk_overlap: Overlap between consecutive chunks to maintain context.
    :return: List of split document chunks.
    """
    try:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Split the text of each document and store in the split_texts list
        split_texts = []
        for doc in documents:
            splits = splitter.split_text(doc.page_content)
            split_texts.extend(splits)

        return split_texts

    except Exception as e:
        raise Exception(f"Error splitting code documents: {e}")

# Function to create a vector store for efficient document retrieval.
def create_vectorstore(repo_path):
    """
    Create and return a vector store (index) from documents in the given repository path.
    
    :param repo_path: Path to the repository.
    :return: A FAISS vector store for efficient similarity search.
    """
    try:
        repo_path = Path(repo_path).resolve(strict=True)

        if not repo_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{repo_path}'")

        # Load the documents from the GitHub repository
        documents = load_github_repo(repo_path)
        print(f"Loaded {len(documents)} documents.")

        # Split the documents into smaller chunks
        data_split = split_code_documents(documents)
        print(f"Split into {len(data_split)} chunks.")

        # Generate embeddings for the text chunks using Bedrock embeddings
        data_embedding = BedrockEmbeddings(
            credentials_profile_name='default',
            model_id='amazon.titan-embed-text-v1'
        )

        # Create an index for similarity search using FAISS
        vectorstore_index_creator = VectorstoreIndexCreator(
            text_splitter=CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=20),
            embedding=data_embedding,
            vectorstore_cls=FAISS
        )

        # Create and return the actual vector store
        vectorstore_wrapper = vectorstore_index_creator.from_documents(documents)
        vectorstore = vectorstore_wrapper.vectorstore

        return vectorstore

    except FileNotFoundError as e:
        raise Exception(f"File not found: {e}")
    except Exception as e:
        raise Exception(f"Error creating vector store: {e}")

# Function to list all sub-files within a repository directory.
def list_sub_files(repo_path):
    """
    List all sub-files in the given GitHub repository directory.
    
    :param repo_path: Path to the repository.
    :return: A list of file paths.
    """
    try:
        repo_path = Path(repo_path).resolve(strict=True)
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{repo_path}'")

        # List all files within the directory
        files = [str(file) for file in repo_path.rglob('*') if file.is_file()]
        return files

    except FileNotFoundError as e:
        raise Exception(f"Directory not found: {e}")
    except Exception as e:
        raise Exception(f"Error listing files: {e}")

# Function to initialize and return a Bedrock LLM (e.g., Mistral model).
def git_llm():
    """
    Initialize and return the BedrockLLM model (Mistral in this case).
    
    :return: BedrockLLM object.
    """
    try:
        return BedrockLLM(
            credentials_profile_name='default',
            model_id='mistral.mistral-large-2402-v1:0',
            model_kwargs={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 100
            }
        )
    except Exception as e:
        raise Exception(f"Error initializing LLM: {e}")

# Function to summarize the content of a file using the LLM.
def summarize_file(file_path):
    """
    Summarize a specific file from the GitHub repository.
    
    :param file_path: Path to the file to summarize.
    :return: Summarized text.
    """
    try:
        file_path = Path(file_path).resolve(strict=True)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: '{file_path}'")

        # Read the file's content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Generate the LLM prompt for summarizing the file
        prompt = f"<s>[INST] summarize the following file: {content} [/INST]"

        # Use the Bedrock LLM to generate a response
        llm = git_llm()
        response = llm.generate(prompts=[prompt])

        # Debugging: Print the full LLM response
        print("LLM Response:", response)

        # Extract the summary text from the LLM response
        if response.generations and response.generations[0]:
            summary = response.generations[0][0].text
            print("Summary backend:", summary)
            return summary
        else:
            raise Exception("No summary text found in the LLM response")

    except FileNotFoundError as e:
        raise Exception(f"File not found error: {e}")
    except Exception as e:
        raise Exception(f"Error summarizing file: {e}")

# Function to create a memory buffer for maintaining conversation context.
def memory_chat():
    llm_data = git_llm()
    # Use ConversationSummaryBufferMemory to summarize and maintain conversation history
    memory = ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=300)
    return memory

# Function to handle conversation with RAG (Retrieval-Augmented Generation) using vector store.
def conversation_rag_chat(input_text, memory, vectorstore):
    #Initialize the conversation chain with memory (for handling context).
    llm_chain_data = git_llm()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    
    # Query the vector store (RAG) using similarity search.
    rag_results = vectorstore.similarity_search(query=input_text, k=5)  # Get top 5 relevant documents.

    # Combine retrieved documents into a single context.
    context = "\n".join([doc.page_content for doc in rag_results])
    
    # Format the input for Mistral AI to include both the user's question and the retrieved context.
    mistral_prompt = f"<s>[INST] Using the following context:\n{context}\n\nAnswer the question: {input_text} [/INST]"
    
    # Call Mistral AI to generate a detailed response based on the retrieved context.
    mistral_llm = git_llm()  # Initialize the Mistral model.
    mistral_response = mistral_llm.generate(prompts=[mistral_prompt])
    
    # Extract the text response from the `generations` attribute.
    detailed_answer = mistral_response.generations[0][0].text if mistral_response.generations else "No answer provided."
    
    # Optionally, use the conversation memory to add context and responses.
    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(detailed_answer)

    # Return the final response from Mistral (RAG + AI answer).
    return detailed_answer