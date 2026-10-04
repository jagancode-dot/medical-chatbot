
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def main():
    st.title("MediBot - Medical Information Assistant")
    st.warning(
        "This chatbot provides educational information only. "
        "It does not replace a qualified healthcare professional. "
        "For emergencies, seek immediate medical care."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        custom_prompt = """
Use only the information provided in the context to answer.
If the answer is not in the context, say you don't know.
Do not invent medical information.

Context: {context}
Question: {question}

Answer directly.
"""

        try:
            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:
                st.error("GOOGLE_API_KEY is missing from the .env file.")
                st.stop()

            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash",
                    temperature=0.0,
                    google_api_key=api_key
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt(custom_prompt)
                }
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            sources = response["source_documents"]
            source_text = "\n".join(
                sorted({
                    f"- {doc.metadata.get('source', 'Unknown source')}, "
                    f"page {doc.metadata.get('page', 'Unknown')}"
                    for doc in sources
                })
            )

            result_to_show = result

            if source_text:
                result_to_show += "\n\n**Sources:**\n" + source_text

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append(
                {"role": "assistant", "content": result_to_show}
            )

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

