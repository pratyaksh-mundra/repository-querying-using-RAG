import streamlit as st
import backend as demo  # Replace with your actual backend module
from pathlib import Path

# Configure the Streamlit page
st.set_page_config(page_title="GitHub Repo Interaction with RAG")

# Set the title of the page with custom styling
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Code Repository with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Initialize session state variables to maintain data across interactions
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None  # Will hold the vector store (index) for the repo
if 'memory' not in st.session_state:
    st.session_state.memory = None  # Will hold the conversation memory for the chatbot
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Will store the chat history for displaying conversation

# Input field for the user to enter the path to the GitHub repository
repo_path = st.text_input("Enter the path to the GitHub repository:")

# Check if a repository path has been provided
if repo_path:
    # If the vector index is not already created, create it
    if st.session_state.vector_index is None:
        with st.spinner("📀 Loading repository and creating vector store..."):
            try:
                # Create and store the vector store from the repository
                st.session_state.vector_index = demo.create_vectorstore(repo_path)
                st.success("Vector store created successfully!")
            except Exception as e:
                st.error(f"Error loading repository: {e}")

    # Dropdown menu to choose an action
    option = st.selectbox(
        "Choose an action:",
        ["Select an action", "Summarize a specific file", "Query the entire repository"]
    )

    # Option to summarize a specific file
    if option == "Summarize a specific file":
        if repo_path:
            try:
                # Get a list of all sub-files in the repository
                file_options = demo.list_sub_files(repo_path)
                # Dropdown menu to select a file to summarize
                file_to_summarize = st.selectbox("Select a file to summarize:", file_options)

                # Button to trigger the summarization process
                if file_to_summarize:
                    if st.button("Summarize"):
                        with st.spinner("📋 Summarizing file..."):
                            try:
                                # Summarize the selected file using the backend function
                                llm_response = demo.summarize_file(file_to_summarize)
                                
                                # Display the summary if available
                                if llm_response:
                                    st.markdown("### File Summary")
                                    st.write(llm_response)
                                else:
                                    st.warning("No summary provided.")
                                    
                            except Exception as e:
                                st.error(f"Error summarizing file: {e}")

            except Exception as e:
                st.error(f"Error listing files: {e}")

    # Option to query the entire repository
    elif option == "Query the entire repository":
        if repo_path:
            if st.session_state.memory is None:
                # Initialize conversation memory if not already initialized
                st.session_state.memory = demo.memory_chat()

            # Display previous chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["text"])

            # Input field for the user to ask a question to the chatbot
            input_text = st.chat_input("Ask a question to the chatbot:")

            if input_text:
                # Display the user's question
                with st.chat_message("user"):
                    st.markdown(input_text)

                # Add user input to chat history
                st.session_state.chat_history.append({"role": "user", "text": input_text})

                try:
                    # Get the response from the chatbot using the RAG approach
                    chat_response = demo.conversation_rag_chat(
                        input_text,
                        st.session_state.memory,
                        st.session_state.vector_index
                    )
                except Exception as e:
                    chat_response = f"An error occurred: {str(e)}"

                # Display the assistant's response
                with st.chat_message("assistant"):
                    st.markdown(chat_response)

                # Add assistant's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

        else:
            st.warning("Please enter the path to the GitHub repository.")

else:
    st.warning("Please enter the path to the GitHub repository.")
