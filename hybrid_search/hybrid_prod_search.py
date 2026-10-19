from langchain_chroma import chroma 
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv 

load_dotenv()

embeddings =  GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.",
        metadata={"type": "product"}
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"}
    ),
    Document(
        page_content="For network connectivity issues, first check the ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"}
    ),
    Document(
        page_content="Authentication error E_AUTH_401 indicates invalid credentials. Verify your username, password, and API token.",
        metadata={"type": "auth"}
    ),
    Document(
        page_content="To configure a static IP address, open network settings and enter the IP address, subnet mask, gateway, and DNS server.",
        metadata={"type": "config"}
    ),
    Document(
        page_content="All network devices must comply with the organization's security policy, including approved firmware and encrypted management connections.",
        metadata={"type": "compliance"}
    ),
    Document(
        page_content="Product SKU-8821A is a dual-band Wi-Fi 6 router with four gigabit LAN ports and WPA3 security.",
        metadata={"type": "product"}
    ),
    Document(
        page_content="Error code E_TIMEOUT_504 means the server did not respond within the allowed time limit.",
        metadata={"type": "error"}
    ),
    Document(
        page_content="If your router frequently disconnects, restart it, inspect the cables, and check whether its firmware is up to date.",
        metadata={"type": "troubleshooting"}
    ),
    Document(
        page_content="Administrator access requires multi-factor authentication. Never share passwords, API keys, or recovery codes with unauthorized users.",
        metadata={"type": "auth"}
    ),
]

print(f'Loaded {len(documents)} documents')