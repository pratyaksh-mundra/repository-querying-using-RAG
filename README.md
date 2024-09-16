# repository-querying-using-RAG

# Code Repository with Retrieval-Augmented Generation (RAG) 🎯

## Overview

This project allows you to interact with a GitHub code repository using a Retrieval-Augmented Generation (RAG) system. The system can summarize code files and answer questions about the repository content by leveraging embeddings and vector search techniques.

## Features

- **Summarize Code Files**: Summarize specific files within a GitHub repository.
- **Query the Repository**: Ask questions about the entire repository using an interactive chatbot.
- **Vector Store Creation**: Automatically creates a vector store from the repository content for fast document retrieval.

## Setup

### Prerequisites

- Python 3.7+
- Required Python packages listed in `requirements.txt` (`langchain`, `FAISS`, `streamlit`, etc.)

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo-name.git
    ```

2. Navigate to the project directory:
    ```bash
    cd your-repo-name
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Streamlit App

1. Run the following command to start the Streamlit app:
    ```bash
    streamlit run frontend.py
    ```

2. You will see the web interface where you can enter the GitHub repository path and choose actions like summarizing files or querying the repository.

### Frontend Features

- **Enter Repository Path**: Enter the path to the GitHub repository you want to analyze.
- **Summarize Specific File**: Select a file from the repository to summarize its content.
- **Query the Repository**: Ask questions about the repository, and the chatbot will respond with context-aware answers based on the documents.

## Example Workflow

1. **Provide the Repository Path**: Enter the path of the GitHub repository you want to explore.
   
2. **Summarize a Specific File**: Choose a file from the list, click "Summarize," and the system will generate a concise summary of the file's contents.
   
3. **Query the Repository**: Ask a question about the repository's code or documentation, and the system will retrieve relevant information and answer based on the file content.

## Contributing

Feel free to fork this project and make contributions! You can create issues for any bugs or features you'd like to see improved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
