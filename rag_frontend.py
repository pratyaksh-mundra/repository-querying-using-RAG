import streamlit as st
import test as demo # replace with your backend filename
from pathlib import Path

st.set_page_config(page_title="GitHub Repo Interaction with RAG")

# Title
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Code Repository with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Initialize session state variables
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input for directory
repo_path = st.text_input("Enter the path to the GitHub repository:")

if repo_path:
    if st.session_state.vector_index is None:
        with st.spinner("📀 Loading repository and creating vector store..."):
            try:
                # Load the vector store from the repo
                st.session_state.vector_index = demo.create_vectorstore(repo_path)
                st.success("Vector store created successfully!")
            except Exception as e:
                st.error(f"Error loading repository: {e}")

    # Dropdown menu for user choices
    option = st.selectbox(
        "Choose an action:",
        ["Select an action", "Summarize a specific file", "Query the entire repository"]
    )

    # Summarize a specific file option
    if option == "Summarize a specific file":
        if repo_path:
            try:
                # List all the sub-files in the repository
                file_options = demo.list_sub_files(repo_path)
                file_to_summarize = st.selectbox("Select a file to summarize:", file_options)

                if file_to_summarize:
                    # Add a button to trigger summarization
                    if st.button("Summarize"):
                        with st.spinner("📋 Summarizing file..."):
                            try:
                                # Summarize the selected file
                                llm_response = demo.summarize_file(file_to_summarize)
                                
                                # Display the summary
                                if llm_response:
                                    st.markdown("### File Summary")
                                    st.write(llm_response)
                                else:
                                    st.warning("No summary provided.")
                                    
                            except Exception as e:
                                st.error(f"Error summarizing file: {e}")

            except Exception as e:
                st.error(f"Error listing files: {e}")


    # Query the entire repository option
    elif option == "Query the entire repository":
        if repo_path:
            if st.session_state.memory is None:
                # Initialize memory for the chatbot
                st.session_state.memory = demo.memory_chat()

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["text"])

            # Chat input for querying the repository
            input_text = st.chat_input("Ask a question to the chatbot:")

            if input_text:
                # Display the user input
                with st.chat_message("user"):
                    st.markdown(input_text)

                st.session_state.chat_history.append({"role": "user", "text": input_text})

                try:
                    # Get the response from the chatbot using RAG
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

                st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

        else:
            st.warning("Please enter the path to the GitHub repository.")

else:
    st.warning("Please enter the path to the GitHub repository.")
