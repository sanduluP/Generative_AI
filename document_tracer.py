import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="TrustifAI", layout="wide")

st.markdown("""
## Welcome to TrustifAI""")

col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox(
        "Select Embedding Provider:",
        ["Hugging Face", "Google", "OpenAI"],
        key="provider_select"
    )

with col2:
    if provider == "Google":
        model_name = st.selectbox(
            "Select Model Name:",
            ["models/embedding-001", "Model 2 (Google Example)"],  
            key="google_model_select"
        )
    elif provider == "Hugging Face":
        model_name = st.selectbox(
            "Select Model Name:",
            ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"],  
            key="huggingface_model_select"
        )
    elif provider == "OpenAI":
        model_name = st.selectbox(
            "Select Model Name:",
            ["davinci", "curie", "babbage", "ada"],  
            key="openai_model_select"
        )

api_key_label = f"Enter your {provider} API Key:"
api_key = st.text_input(api_key_label, type="password", key="api_key_input")

chunking_option = st.selectbox(
    "Select Chunking Method:",
    ["Text Chunking", "Sentence Chunking"]
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_sentence_chunks(text):
    sentences = sent_tokenize(text)
    # Optionally, you can group sentences into chunks if needed
    return sentences
    
def get_vector_store(chunks, api_key):
    if provider == "Hugging Face":
        # HuggingFaceEmbedding
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name=model_name)
        vector_store = Chroma.from_documents(chunks,embeddings)
        vector_store.persist()
    # OpenAIEmbedding
    elif provider == "OpenAI":
        pass
    #GoogleEmbedding
    elif provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    if provider == "Google":
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    elif provider == "Hugging Face":
        model = HuggingFaceHub(model_name=model_name, huggingfacehub_api_token=api_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    if provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    elif provider == "Hugging Face":
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name=model_name)
        new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)  # Assuming you use Chroma for Hugging Face
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                
                if chunking_option == "Text Chunking":
                    chunks = get_text_chunks(raw_text)
                elif chunking_option == "Sentence Chunking":
                    chunks = get_sentence_chunks(raw_text)
                    
                get_vector_store(chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()
