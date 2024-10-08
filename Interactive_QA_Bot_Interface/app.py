import streamlit as st
import numpy as np
from src.components.read_pdf_file import PDFFileReader
from src.components.document_embedding import DocumentEmbedding
from src.components.faiss_indexing import FAISSIndexing
from src.components.generate_response import GenerateResponse
from src.components.retrive_relevent_docs import RetriveReleventDocs
from src.components.spliting_into_chunks import SplitIntoChunks

# Initialize components
PDFFileReader = PDFFileReader()
DocumentEmbedding = DocumentEmbedding()
FAISSIndexing = FAISSIndexing()
GenerateResponse = GenerateResponse()
RetriveReleventDocs = RetriveReleventDocs()
SplitIntoChunks = SplitIntoChunks()

# Streamlit Interface
st.title("Interactive QA Bot with Document Upload")
st.write("Upload up to 3 PDF documents and ask questions based on their content.")

# Display instructions for the user
st.markdown("""
### Instructions:
1. **Upload PDF Files**: You can upload up to 3 PDF files.
2. **Ask a Question**: After the documents are processed, input a question based on their content.
3. **View Answer**: The bot will provide a response based on the content from the uploaded files.
4. **See Relevant Document Segments**: The relevant document segments are displayed with similarity scores.
""")

# Button to start a new conversation
if st.button("Start New Conversation"):
    st.session_state.conversation_history = []
    st.session_state.conversation_count = 0
    st.session_state.selected_chunk = None
    st.session_state.combined_context = ""
    st.success("Conversation reset. You can now start a new conversation.")

# File upload section
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("You can only upload up to 5 files. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:3]

    with st.spinner('Processing documents...'):
        all_chunks = []
        all_embeddings = []

        for uploaded_file in uploaded_files:
            # Extract text from the uploaded PDF
            document_text = PDFFileReader.extract_text_from_pdf(uploaded_file)

            # Split the text into smaller chunks for embedding
            chunks = SplitIntoChunks.split_text(document_text)
            all_chunks.extend(chunks)

            # Generate embeddings for each chunk
            chunk_embeddings = DocumentEmbedding.create_embeddings(chunks)
            all_embeddings.extend(chunk_embeddings)

        # Build FAISS index with chunk embeddings from all documents
        index = FAISSIndexing.build_faiss_index(all_embeddings)

        st.success("Documents processed successfully! You can now ask questions.")

        # Initialize conversation history in session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'selected_chunk' not in st.session_state:
            st.session_state.selected_chunk = None

        if 'combined_context' not in st.session_state:
            st.session_state.combined_context = ""

        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0

        # Sidebar for conversation history
        st.sidebar.title("Conversation History")
        if st.session_state.conversation_history:
            for i, item in enumerate(st.session_state.conversation_history):
                st.sidebar.markdown(f"**User Query {i+1}:** {item['query']}")
                st.sidebar.markdown(f"**Response {i+1}:** {item['response']}\n")

        # User query section
        user_query = st.text_input("Ask a question about the documents:")

        if user_query:
            with st.spinner('Retrieving relevant information...'):
                # Retrieve relevant chunks from the documents
                relevant_chunks = RetriveReleventDocs.retrieve(user_query, index, all_chunks)

                # Combine relevant chunks into context for initial response
                if relevant_chunks:
                    combined_context = " ".join([chunk for chunk, _ in relevant_chunks])
                    st.session_state.combined_context = combined_context

                    # Display the initial answer based on all relevant segments
                    with st.spinner('Generating answer from all relevant segments...'):
                        prompt = f"""
                        **Context/Knowledge**: {combined_context} \n\n 
                        **Query**: {user_query} \n\n 
                        **Instruction**: If you do not have enough information to answer the question, clearly state that you cannot provide an answer based on the current context. Do not fabricate or make up information.
                        **Response** : 
                        """
                        initial_response = GenerateResponse.generate_response_from_prompt(prompt)
                        st.session_state.initial_response = initial_response
                        st.markdown(f"**Response:** {st.session_state.initial_response}")

                    # Display relevant document segments
                    st.subheader("Relevant Document Segments:")
                    for i, (chunk, score) in enumerate(relevant_chunks):
                        st.markdown(f"**Segment {i+1}:** (Relevance Score: {score:.4f})")
                        st.markdown(chunk)

                else:
                    st.session_state.initial_response = "No relevant information found in the documents."
                    st.markdown(f"**Response:** {st.session_state.initial_response}")

            # Manage conversation history and count
            if st.session_state.conversation_count >= 30:
                st.warning("The current conversation length exceeds the limit of 30 exchanges. Restarting the conversation.")
                st.session_state.conversation_history = []
                st.session_state.conversation_count = 0  # Reset conversation count

            # Add current user query and response to conversation history
            if user_query and 'initial_response' in st.session_state:
                st.session_state.conversation_history.append({'query': user_query, 'response': st.session_state.initial_response})
                st.session_state.conversation_count += 1  # Increment conversation count
