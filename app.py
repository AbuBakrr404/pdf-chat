import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastembed import TextEmbedding
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import tempfile, os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and "vectorstore" not in st.session_state:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        os.unlink(tmp_path)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

if "vectorstore" in st.session_state:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""Answer based only on the context below.

Context: {context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.markdown("**Answer:**")
        st.write(answer)