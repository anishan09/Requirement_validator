# retrival_fn.py

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI

# Load HuggingFace Embedding model (same as chunking)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Chroma vector store
vectorstore = Chroma(
    persist_directory="chroma_db_incose",
    embedding_function=embedding_model
)

# Set up retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # get diverse relevant chunks
    search_kwargs={"k": 5}
)

# Set up GPT-4o LLM from OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Core function to check validity of requirement
def check_requirement_validity(requirement_text: str) -> str:
    # Step 1: Retrieve top relevant chunks from Chroma
    relevant_docs = retriever.get_relevant_documents(requirement_text)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Step 2: Create prompt to evaluate requirement
    prompt = f"""
You are a systems engineering expert. Evaluate the following requirement based on INCOSE standards.

Context (from INCOSE document):
{context}

Requirement to evaluate:
"{requirement_text}"

Task:
1. Determine whether the requirement is VALID or INVALID.
2. Use INCOSE criteria such as clarity, feasibility, testability, traceability, etc.
3. Justify your conclusion and suggest improvements if invalid.

Format:
---
Result: VALID / INVALID
Reason: <Detailed explanation>
---
"""

    # Step 3: Get GPT-4o response
    response = llm.invoke(prompt)
    return response.content
