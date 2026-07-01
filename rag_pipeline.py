from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

def load_and_split(file_path: str):
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

def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def build_qa_chain(vectorstore):
    prompt = PromptTemplate.from_template("""You are a helpful assistant. Answer the question
using ONLY the context provided below. If the answer is not in the context,
say "I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:""")

    llm = ChatOllama(model="llama3.1", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def answer_question(chain, retriever, question: str) -> dict:
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.page_content[:200] for doc in sources]
    }

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    chain, retriever = build_qa_chain(vectorstore)
    question = input("Ask a question about the document: ")
    result = answer_question(chain, retriever, question)
    print("\nAnswer:", result["answer"])
    print("\n--- Sources used ---")
    for i, src in enumerate(result["sources"], 1):
        print(f"[{i}] {src[:150]}...")