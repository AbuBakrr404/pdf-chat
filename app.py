import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile, os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and "retriever" not in st.session_state:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = 3

        st.session_state.retriever = retriever
        os.unlink(tmp_path)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

if "retriever" in st.session_state:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = PromptTemplate.from_template("""Answer based only on the context below.

Context: {context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.markdown("**Answer:**")
        st.write(answer)