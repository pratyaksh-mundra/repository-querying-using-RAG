import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
from pathlib import Path
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import BedrockLLM
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from pathlib import Path
import nltk
from langchain.prompts import PromptTemplate

nltk.download('punkt')


def load_github_repo(repo_path, file_types=['.py', '.md', '.sh']):
    """
    Load all files from a given GitHub repository directory.
    """
    try:
        loader = DirectoryLoader(repo_path, glob='**/*.md')
        all_documents = loader.load()

        filtered_documents = [doc for doc in all_documents if any(doc.metadata['source'].endswith(ext) for ext in file_types if ext != '.sh')]

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

def split_code_documents(documents, chunk_size=200, chunk_overlap=20):
    """
    Split code documents into smaller chunks using CharacterTextSplitter.
    """
    try:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        split_texts = []
        for doc in documents:
            splits = splitter.split_text(doc.page_content)
            split_texts.extend(splits)

        return split_texts

    except Exception as e:
        raise Exception(f"Error splitting code documents: {e}")

def create_vectorstore(repo_path):
    """
    Create and return a vector store (index) from documents in the given repository path.
    """
    try:
        repo_path = Path(repo_path).resolve(strict=True)

        if not repo_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{repo_path}'")

        documents = load_github_repo(repo_path)
        print(f"Loaded {len(documents)} documents.")

        data_split = split_code_documents(documents)
        print(f"Split into {len(data_split)} chunks.")

        data_embedding = BedrockEmbeddings(
            credentials_profile_name='default',
            model_id='amazon.titan-embed-text-v1'
        )

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

    
def list_sub_files(repo_path):
    """
    List all sub-files in the given GitHub repository directory.
    """
    try:
        repo_path = Path(repo_path).resolve(strict=True)
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{repo_path}'")

        files = [str(file) for file in repo_path.rglob('*') if file.is_file()]
        return files

    except FileNotFoundError as e:
        raise Exception(f"Directory not found: {e}")
    except Exception as e:
        raise Exception(f"Error listing files: {e}")
    
def git_llm():
    """
    Initialize and return the BedrockLLM model.
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
    
def summarize_file(file_path):
    """
    Summarize a specific file from the GitHub repository.
    """
    try:
        file_path = Path(file_path).resolve(strict=True)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        prompt = f"<s>[INST] summarize the following file: {content} [/INST]"

        llm = git_llm()
        response = llm.generate(prompts=[prompt])

        # Debugging print statements
        print("LLM Response:", response)

        # Access the text from the first generation object in the nested lists
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


    
def memory_chat():
    llm_data = git_llm()
    memory=ConversationSummaryBufferMemory(llm=llm_data,max_token_limit=300)
    return memory      

def conversation_rag_chat(input_text, memory, vectorstore):
    # Step 1: Initialize the conversation chain with memory (for handling context).
    llm_chain_data = git_llm()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    
    # Step 2: Query the vector store (RAG) using similarity search.
    rag_results = vectorstore.similarity_search(query=input_text, k=5)  # Get top 5 relevant documents.

    # Combine retrieved documents into a single context.
    context = "\n".join([doc.page_content for doc in rag_results])
    
    # Step 3: Format the input for Mistral AI to include both the user's question and the retrieved context.
    mistral_prompt = f"<s>[INST] Using the following context:\n{context}\n\nAnswer the question: {input_text} [/INST]"
    
    # Step 4: Call Mistral AI to generate a detailed response based on the retrieved context.
    mistral_llm = git_llm()  # Initialize the Mistral model.
    mistral_response = mistral_llm.generate(prompts=[mistral_prompt])
    
    # Extract the text response from the `generations` attribute.
    detailed_answer = mistral_response.generations[0][0].text if mistral_response.generations else "No answer provided."
    
    # Step 5: Optionally, use the conversation memory to add context and responses.
    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(detailed_answer)

    # Step 6: Return the final response from Mistral (RAG + AI answer).
    return detailed_answer
