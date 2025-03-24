import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

# llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key=SecretStr("ollama"),
#     model="mistral",
#     temperature=0,
# )

# think_llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key=SecretStr("ollama"),
#     model="qwq",
# )


llm = ChatOpenAI(api_key=SecretStr(os.getenv("openai_key", "")), model="gpt-4o-mini")

think_llm = ChatOpenAI(api_key=SecretStr(os.getenv("openai_key", "")), model="gpt-4o-mini")
