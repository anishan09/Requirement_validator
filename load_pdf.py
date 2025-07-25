#just demo of how to upload pdf before doing actual coding

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("incose_std.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages.")
print(docs[0].page_content[:500])
