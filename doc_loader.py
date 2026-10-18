from google.genai import documents
from dotenv import load_dotenv
import os
import tempfile 
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader, 
    DirectoryLoader,
    PyPDFLoader
    )

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False,suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\nThis file is used to test the doc loader file")
        temp_file_path = temp_file.name


    try : 

        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        print("Document Content:")
        print(documents[0].page_content)
        print("Metadata:")
        print(documents[0].metadata)

    finally : 
        #clean up the temporary files 
        if os.path.exists(temp_file_path) :
            os.remove(temp_file_path)
        print("cleaned up")

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Page: {doc.metadata['page']}")
        print(f"Content: {doc.page_content[:100]}...")  


if __name__ == "__main__":
    pdf_loader("./docs/Krishna_Gaur_LOR.pdf")