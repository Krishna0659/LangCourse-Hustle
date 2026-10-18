import os
import tempfile

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


# ============================================================
# 2. INITIALIZE EMBEDDINGS
# ============================================================

# Gemini embeddings convert text into vector representations.
# These vectors are stored and searched using ChromaDB.

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# 3. INITIALIZE THE CHAT MODEL
# ============================================================

if LLM_PROVIDER == "gemini":

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing from your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

elif LLM_PROVIDER == "openrouter":

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is missing from your .env file."
        )

    llm = ChatOpenAI(
        model="mistralai/mistral-small-latest",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
    )

else:
    raise ValueError(
        "Invalid LLM_PROVIDER. Choose 'gemini' or 'openrouter'."
    )


# ============================================================
# 4. SAMPLE KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by
language models. It was created by Harrison Chase in October 2022.

## Core Components

1. Models: LangChain supports various LLM providers including
OpenAI, Anthropic, and local models.

2. Prompts: Templates for structuring inputs to language models.

3. Chains: Sequences of calls to models and other components.

4. Agents: Systems that use LLMs to determine which actions to take.

5. Memory: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications.
Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free.
LangSmith has a free tier and paid plans.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


# ============================================================
# 5. CREATE KNOWLEDGE BASE / VECTOR STORE
# ============================================================

def create_kb():
    """Create a Chroma vector store from the knowledge base."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={
            "source": "langchain_knowledge_base.md"
        },
    )

    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store


# ============================================================
# 6. BASIC RAG DEMO
# ============================================================

def demo_basic_rag():

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Answer concisely. If you don't know the answer,
say "I don't know."
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("\n========== BASIC RAG DEMO ==========\n")

    for question in questions:
        answer = rag_chain.invoke(question)

        print(f"Q: {question}")
        print(f"A: {answer}\n")


# ============================================================
# 7. RAG WITH SOURCES
# ============================================================

def demo_rag_with_sources():

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using the context below.

Include the source names in your answer.
Do not invent sources.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs_with_sources(docs):

        formatted = []

        for i, doc in enumerate(docs):

            source = doc.metadata.get(
                "source",
                "unknown",
            )

            formatted.append(
                f"[{i + 1}] Source: {source}\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What are the core components of LangChain?"

    answer = rag_chain.invoke(question)

    print("\n========== RAG WITH SOURCES ==========\n")
    print(f"Q: {question}")
    print(f"A: {answer}")


# ============================================================
# 8. RAG WITH FALLBACK
# ============================================================

def demo_rag_with_fallback():

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer based ONLY on the context below.

If the answer cannot be found in the context,
respond exactly with:
"I don't have information about that in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is the pricing for LangSmith?",
        "What is the stock price of OpenAI?",
        "How do I deploy LangChain to AWS?",
    ]

    print("\n========== RAG WITH FALLBACK ==========\n")

    for question in questions:

        answer = rag_chain.invoke(question)

        print(f"Q: {question}")
        print(f"A: {answer}\n")


# ============================================================
# 9. STRUCTURED RAG
# ============================================================

def demo_structured_rag():
    """RAG with structured output."""

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    class RAGResponse(BaseModel):

        answer: str = Field(
            description="The answer to the question"
        )

        confidence: str = Field(
            description="Confidence level: high, medium, or low"
        )

        sources_used: List[str] = Field(
            description="List of source names referenced"
        )

        follow_up: str = Field(
            description="A suggested follow-up question"
        )

    structured_llm = llm.with_structured_output(
        RAGResponse
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the provided context.

Use only the provided context.
List only sources that appear in the context.

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):

        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | structured_llm
    )

    result = rag_chain.invoke(
        "What is LangGraph?"
    )

    print("\n========== STRUCTURED RAG ==========\n")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")


# ============================================================
# 10. DOCUMENT Q&A EXERCISE
# ============================================================

def exercise_document_qa():
    """
    Build a document Q&A system that:
    1. Accepts a text document.
    2. Splits and embeds the document.
    3. Supports multiple questions.
    4. Returns answers with confidence labels.
    """

    class DocumentQA:

        def __init__(
            self,
            document: str,
            source_name: str = "document",
        ):

            # Split the input document into chunks.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )

            doc = Document(
                page_content=document,
                metadata={
                    "source": source_name
                },
            )

            chunks = splitter.split_documents([doc])

            # Create a vector store.
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
            )

            # Create a retriever.
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            # Use the configured Gemini/OpenRouter model.
            self.llm = llm

            # Create the prompt.
            self.prompt = ChatPromptTemplate.from_template(
                """
Answer the question using the provided context only.

If the answer is not present in the context, say:
"I don't have enough information in the document."

Give a confidence label of high, medium, or low.
The confidence label is your assessment, not a calibrated probability.

Context:
{context}

Question:
{question}

Format:
[Confidence: high/medium/low] Answer
"""
            )

            def format_docs(docs):

                return "\n\n".join(
                    doc.page_content for doc in docs
                )

            # Build the RAG chain.
            self.chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

        def ask(self, question: str) -> str:
            return self.chain.invoke(question)

    # Test document.
    test_doc = """
    The Python programming language was created by Guido van Rossum.
    First released in 1991, Python emphasizes code readability.
    Python 3.12 was released in October 2023 with improved error messages.
    The language is named after Monty Python, not the snake.
    """

    qa = DocumentQA(
        document=test_doc,
        source_name="python_facts",
    )

    questions = [
        "Who created Python?",
        "When was Python 3.12 released?",
        "Why is Python named Python?",
    ]

    print("\n========== DOCUMENT Q&A SYSTEM ==========\n")

    for question in questions:

        answer = qa.ask(question)

        print(f"Q: {question}")
        print(f"A: {answer}\n")


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    # Uncomment ONE demo at a time.

    demo_basic_rag()
    # demo_rag_with_sources()
    # demo_rag_with_fallback()
    # demo_structured_rag()

    #exercise_document_qa()