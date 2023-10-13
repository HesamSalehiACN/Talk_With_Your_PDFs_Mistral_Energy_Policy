import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain import PromptTemplate

HUGGINGFACEHUB_API_TOKEN = "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"

# Function for reading PDFs and converting them to a plain text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function for splitting the plain text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function for embedding the text and save it in a vectorstore
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function for creating defining llm, allocating memory, and creating a conversation chain
def get_conversation_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                         repo_id=repo_id,
                         model_kwargs={"temperature":0.1, "max_new_tokens":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    template = """Consider the following context to generate a response for the question below.
                  Your role is a Consultant/Advisor to retrieve information from the documents, Offer strategic insights,
                  policy advice, or recommendations based on the document's content.
                  Given the context and the role, provide a clear, concise, and comprehensive answer
                  using a natural language tone.
                  The documents are related to the energy industry.
                  {context}
                  
                  Question: {question}
                  Recommended Answer:"""
    
    chain_prompt = PromptTemplate.from_template(template)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="map_reduce"
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chain_prompt}
    )
    return conversation_chain


# Function for handling the user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# Building the main page with its sidebar by Streamlit
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDF documents")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Header of the main page
    st.header("Talk with your documents - Mistral")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.image("docs/Accenture_logo_PNG1.png", use_column_width=True)

        # st.sidebar.markdown('<br>', unsafe_allow_html=True)
        
        #st.subheader("OpenAI key")
        #openai_api_key = st.sidebar.text_input('OpenAI API Key')
        #if openai_api_key:
             # Display the entered key on the sidebar
             #st.text(f"Entered OpenAI API Key: {openai_api_key}")

        st.subheader("Upload your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Upload'", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
