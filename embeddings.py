# embeddings.py

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json
import os

# Load the chunked text (from your previous chunking step)
with open("chunked_incose.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert chunks to LangChain Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Initialize HuggingFace Embeddings model (Free, local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize and persist Chroma vector DB
persist_dir = "chroma_db_incose"  # You can rename this if needed

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_dir
)

# Save to disk
vectorstore.persist()

print("✅ Embeddings stored successfully in Chroma DB at:", persist_dir)
