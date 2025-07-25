#step 1
# Here we uploaded the pdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("incose_std.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages.")
print(docs[0].page_content[:500])

#step 2
# here we started using RecursiveCharacterTextSplitter for chunking purpose
# Set the chunk size and overlap

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split into chunks
chunks = text_splitter.split_documents(docs)

#step 3
#lets inspect chunks
#This will confirm:
# 1. Sentences aren't being cut mid-way
# 2. Semantic structure is preserved
# 3. Paragraphs/sections are maintained

for i, chunk in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
    print()


#step 4 
#save these chunks for further use (to make embeddings)

import json
with open("chunked_incose.json", "w", encoding="utf-8") as f:
    json.dump([doc.page_content for doc in chunks], f, ensure_ascii=False, indent=2)
