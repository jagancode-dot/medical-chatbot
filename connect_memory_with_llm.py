
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Setup Gemini
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.5,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

CUSTOM_PROMPT_TEMPLATE = """
Use the information in the context to answer the user's question.
If the answer is not available in the context, say you don't know.
Do not make up medical information.

Context: {context}
Question: {question}

Answer directly.
"""

def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# Create question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])