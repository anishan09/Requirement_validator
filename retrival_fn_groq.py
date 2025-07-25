# retrival_fn_groq.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup embedding model and load Chroma DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db_incose", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Groq LLM client
client = Groq(api_key=GROQ_API_KEY)

def check_requirement_validity(requirement_text: str, model_name: str) -> str:
    """
    Evaluate whether the input requirement is valid using the selected LLM model.
    
    Args:
        requirement_text (str): The requirement statement to validate.
        model_name (str): Model name to use (e.g., 'llama3-70b-8192', 'kimi-k2', 'gemma-2b-it').
    
    Returns:
        str: The LLM's decision and reasoning.
    """
    # Step 1: Retrieve relevant context
    relevant_docs = retriever.get_relevant_documents(requirement_text)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Step 2: Construct prompt
    prompt = f"""
You are a systems engineering expert familiar with INCOSE standards.

Context:
{context}

Requirement:
"{requirement_text}"

Task:
1. Determine whether the requirement is VALID or INVALID based on the INCOSE context.
2. Justify your decision using key parameters such as clarity, completeness, feasibility, verifiability, and traceability.
3. Format the response as:
---
Result: VALID / INVALID
Reason: <detailed reasoning>
---
"""

    # Step 3: Get response from the selected Groq-hosted LLM
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content
