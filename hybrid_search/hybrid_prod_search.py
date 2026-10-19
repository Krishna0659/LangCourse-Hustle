import os
from typing import List
from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv 

load_dotenv()

# MANUAL WEIGHTED RECIPROCAL RANK FUSION

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion."""

    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)

        for rank, doc in enumerate(results):
            # Use page content as the document identifier
            key = doc.page_content

            # Weighted RRF score
            rrf_score = weight * (1.0 / (rank + rrf_k))

            if key in doc_scores:
                doc_scores[key] = (
                    doc_scores[key][0] + rrf_score,
                    doc
                )
            else:
                doc_scores[key] = (rrf_score, doc)

    # Sort documents by descending RRF score
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in sorted_docs[:k]]


# HYBRID RETRIEVER

class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search."""

    def __init__(
        self,
        documents: List[Document],
        bm25_weight: float = 0.5,
        k: int = 4
    ):

        if not 0 <= bm25_weight <= 1:
            raise ValueError("bm25_weight must be between 0 and 1.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        if not documents:
            raise ValueError("At least one document is required.")

        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        # ----------------------------------------------------
        # Initialize embedding model
        # ----------------------------------------------------

        self.embeddings =  GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2"
        )

        # ----------------------------------------------------
        # Create Chroma vector store
        # ----------------------------------------------------

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="hybrid_search"
        )

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # ----------------------------------------------------
        # Create BM25 retriever
        # ----------------------------------------------------

        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=k
        )

    # SEARCH

    def search(self, query: str) -> List[Document]:
        """Run hybrid search using weighted RRF."""

        return hybrid_retrieve(
            query=query,
            retrievers=[
                self.bm25_retriever,
                self.vector_retriever
            ],
            weights=[
                self.bm25_weight,
                self.vector_weight
            ],
            k=self.k,
            rrf_k=60
        )


    # ADD DOCUMENTS

    def add_documents(self, documents: List[Document]):
        """Add documents to Chroma and rebuild the BM25 retriever."""

        if not documents:
            return

        # Add new documents to Chroma
        self.vectorstore.add_documents(documents)

        # Retrieve all current documents from Chroma
        all_docs = self.vectorstore.get(
            include=["documents", "metadatas"]
        )

        # Reconstruct LangChain Document objects
        stored_documents = [
            Document(
                page_content=content,
                metadata=metadata or {}
            )
            for content, metadata in zip(
                all_docs["documents"],
                all_docs["metadatas"]
            )
        ]

        # Rebuild BM25 using all documents
        self.bm25_retriever = BM25Retriever.from_documents(
            stored_documents,
            k=self.k
        )


# ============================================================
# MAIN: DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    documents = [
        Document(
            page_content=(
                "Product SKU-7742X is our flagship router. "
                "It supports gigabit speeds and advanced QoS features."
            ),
            metadata={"type": "product"}
        ),

        Document(
            page_content=(
                "Error code E_CONN_REFUSED indicates the server "
                "rejected the connection. Check firewall settings."
            ),
            metadata={"type": "error"}
        ),

        Document(
            page_content=(
                "For network connectivity issues, first check "
                "the ethernet cable and router status lights."
            ),
            metadata={"type": "troubleshooting"}
        ),

        Document(
            page_content=(
                "Authentication error E_AUTH_401 indicates invalid "
                "credentials. Verify your username and API token."
            ),
            metadata={"type": "auth"}
        ),

        Document(
            page_content=(
                "To configure a static IP address, enter the IP "
                "address, subnet mask, gateway, and DNS server."
            ),
            metadata={"type": "config"}
        ),

        Document(
            page_content=(
                "All network devices must comply with the "
                "organization's security policy."
            ),
            metadata={"type": "compliance"}
        ),

        Document(
            page_content=(
                "Product SKU-8821A is a dual-band Wi-Fi 6 router "
                "with four gigabit LAN ports and WPA3 security."
            ),
            metadata={"type": "product"}
        ),

        Document(
            page_content=(
                "Error code E_TIMEOUT_504 means the server did "
                "not respond within the allowed time limit."
            ),
            metadata={"type": "error"}
        ),

        Document(
            page_content=(
                "If your router frequently disconnects, restart it "
                "and check whether its firmware is up to date."
            ),
            metadata={"type": "troubleshooting"}
        ),

        Document(
            page_content=(
                "Administrator access requires multi-factor "
                "authentication. Never share passwords or API keys."
            ),
            metadata={"type": "auth"}
        ),
    ]
    # Initialize hybrid retriever
    hybrid = HybridRetriever(
        documents=documents,
        bm25_weight=0.5,
        k=4
    )

    # Run hybrid search
    query = "What does E_CONN_REFUSED mean?"

    results = hybrid.search(query)

    print(f"\nQuery: {query}")
    print("\nHybrid Search Results:\n")

    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}\n")

    # --------------------------------------------------------
    # Add a new document
    # --------------------------------------------------------

    new_documents = [
        Document(
            page_content=(
                "Error E_DNS_FAILURE indicates that the domain "
                "name could not be resolved by the DNS server."
            ),
            metadata={"type": "error"}
        )
    ]

    hybrid.add_documents(new_documents)

    print("\nNew document added successfully!")

    # Search again after updating the documents
    query = "What causes DNS resolution failures?"

    results = hybrid.search(query)

    print(f"\nQuery: {query}")
    print("\nUpdated Hybrid Search Results:\n")

    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}\n")
