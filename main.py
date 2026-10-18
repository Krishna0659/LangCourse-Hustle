from langchain_openai import ChatOpenAI
from importlib.metadata import version
from dotenv import load_dotenv
import os

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )

    response1 = llm.invoke("Say 'setup complete!' in one word")
    print(f"Google Gemini's response: {response1.content}")

    # OpenRouter
    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )

    response2 = llm.invoke("Say 'setup complete!' in one word")
    print(f"OpenRouter's response: {response2.content}")


if __name__ == "__main__":
    main()