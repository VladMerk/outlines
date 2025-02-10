import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

# llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key=SecretStr("ollama"),
#     model="hf.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF"
#     # model="hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
#     # model="mistral-nemo"
# )

llm = ChatOpenAI(
    api_key=SecretStr(os.getenv('openai_key', "")),
    model="gpt-4o-mini"
)
