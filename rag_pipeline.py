from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(file_path: str):
    """Load document and split into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    chunks = load_and_split("test_document.pdf")
    print(f"{len(chunks)} chunks created")
    print("---- First chunk preview ----")
    print(chunks[0].page_content[:300])