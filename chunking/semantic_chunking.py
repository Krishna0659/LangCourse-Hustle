from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2")

document = '''
#What is LangChain?
LangChain is an open-source framework that simplifies the development of applications powered by large language models (LLMs). It provides a standard interface for connecting LLMs to other components like data sources, APIs, and memory systems. LangChain abstracts away the complexity of working directly with LLM APIs, allowing developers to focus on building high-level application logic.

#Core Concepts
#Components
#Prompt Templates: Pre-defined structures for LLM inputs that can include variables and instructions.
#LLMs: Wrappers around different language models (OpenAI, Hugging Face, etc.).
#Chains: Sequences of calls to LLMs or other utilities.
#Retrievers: Modules for fetching and retrieving documents.
#Agents: Systems that use LLMs to decide which actions to take and in what order.
#Memory: Mechanisms for persisting state between calls to an application.

#Key Features
#Modularity: Components can be combined like building blocks.
#Abstractions: Provides standard interfaces for common LLM tasks.
#Integration: Easy to connect with various data sources and tools.
#Language Support: Available in Python and JavaScript/TypeScript.

#What is LangSmith?
LangSmith is a development platform for building and monitoring LLM-powered applications. It is designed to work seamlessly with LangChain, providing observability into the inner workings of your applications. LangSmith allows you to debug, test, evaluate, and monitor your applications in production.

#Core Features
#Tracing: Visualizes the complete execution flow of your application, including all intermediate steps and LLM calls.
#Debugging: Helps identify issues by showing inputs, outputs, and errors for each component.
#Monitoring: Tracks application performance and usage patterns in real-time.
#Evaluation: Provides tools to test and compare different versions of your application.
#Annotation: Allows manual review and labeling of application outputs.

#How LangChain and LangSmith Work Together?
LangChain handles the orchestration and logic of your application, while LangSmith provides visibility and control over that execution. When you build an application with LangChain and enable LangSmith tracing, every step in your chain is automatically logged to the LangSmith platform. This allows you to:

#Visualize the data flow between components.
#Inspect the exact prompts sent to LLMs and the responses received.
#Track token usage and costs.
#Compare the performance of different model versions.
#Debug issues by replaying specific runs and examining intermediate states.

#Practical Example
#Installation
```bash
pip install langchain langchain-openai langsmith
```

#Code Example
```python
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Initialize LangSmith client (optional - automatic if LANGSMITH_ENDPOINT and LANGSMITH_API_KEY are set)
client = Client()

# Define your LLM and prompt
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nQuestion: {question}\nAnswer:"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with tracing enabled
response = chain.run("What is LangChain?")
print(response)

# The run will be automatically logged to LangSmith
# You can view it in the LangSmith dashboard by following the link provided in the logs
```

#Common Use Cases
#LangChain is used for building:
#Question answering systems
#Chatbots and virtual assistants
#Summarization tools
#Content generation applications
#Data extraction systems

#LangSmith is used for:
#Monitoring production LLM applications
#Debugging agentic systems with complex decision-making
#Evaluating model performance across different datasets
#Managing and versioning prompts
#Collaborating with teams on LLM development

'''
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400 , 
    chunk_overlap = 50 , 
    separators = ['\n\n' , '\n' , '.', ' ']
)

recursive_chunks = recursive_splitter.split_text(document)

semantic_splitter = SemanticChunker(
    embeddings = embeddings ,
    breakpoint_threshold_type = 'percentile',
    breakpoint_threshold_amount = 90  
)
semantic_chunks = semantic_splitter.split_text(document)

recursive_vectorstore = Chroma.from_texts(
    texts=recursive_chunks,
    embedding=embeddings,
    collection_name = 'recursive_splitter'
)


semantic_vectorstore = Chroma.from_texts(
    texts=semantic_chunks,
    embedding=embeddings,
    collection_name = 'semantic_splitter'
)

test_queries = [
    "What is LangChain and how does it simplify LLM application development?",
    "What are the core components of LangChain, including chains, retrievers, agents, and memory?",
    "What is LangSmith, and how does it help developers debug and monitor LLM applications?",
    "How do LangChain and LangSmith work together to provide observability and track token usage?",
    "What are the common use cases of LangChain and LangSmith?"
]

def test_retrieval(query, vectorstore, name):
    results = vectorstore.similarity_search(query, k= 1) 
    print(f'\\n{name} - Query: \"{query}\"')
    print(f'Retrieved : {results[0].page_content[:150]}...')
    return results[0].page_content

print(f"\n{'=' * 60}")
print(f"Retrieval Tests ")
print(f"{'=' * 60}")

for query in test_queries:
    print('=' * 60)
    resursive_results = test_retrieval(query, recursive_vectorstore, 'Recursive Chunking')
    semantic_results = test_retrieval(query, semantic_vectorstore, 'Semantic Chunking')

    print('\nLength Comparison:')
    print(f'Recursive Chunking : {len(resursive_results)} characters')
    print(f'Semantic Chunking  : {len(semantic_results)} characters')
    