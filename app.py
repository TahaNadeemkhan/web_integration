#laoding import libraries
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

#Embedding model
embed_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

import os

#streamlit 
st.set_page_config(page_title="website insights")
st.title("Chat with websites")

#Webloader
def get_vector_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter()
        doc_chunks = text_splitter.split_documents(document)
        
        persist_directory = "chroma_store"
        
        # Create vector store with Chroma
        vector_stores = Chroma.from_documents(doc_chunks, embed_model, persist_directory=persist_directory)
        return vector_stores
    except Exception as e:
        st.error(f"Failed to load website content or create vector store: {str(e)}")
        return None

#creating retriver chain
def context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7)
    retriever = vector_store.as_retriever()
    prompt=ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation ")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


#setting up chain
def get_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You will be provided with a website link.If you cannot find relevant information from the website, inform the user that you can only answer queries related to the site. Only look for answers related to the website's content. \\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', "{input}"),
    ])    
    stuff_chain=create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)


#bot response
def get_response(user_input):
    retriever_chain=context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
    if not response['answer'] or response['answer'].strip() == "":
        return "I can only provide answers related to the content of the website. Please ask something relevant to the site."

    return response['answer']


#sidebar
with st.sidebar:
    st.header("Settings")
    web_url=st.text_input("Enter Website Url")
    button = st.button("Submit", disabled=True)
if web_url is None or web_url=="":
    st.info("Please enter a website URL")
else:  
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="How can i help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_url(web_url)


    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        if isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)
    