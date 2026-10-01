import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "sample_nutrition_docs.json"
CHROMA_DIR = ROOT / "chroma_db"


class Doc(TypedDict):
    id: str
    topic: str
    text: str


def load_docs() -> list[Doc]:
    with open(DATA_PATH) as f:
        return json.load(f)


def get_llm(temperature: float = 0.0) -> ChatAnthropic:
    """
    Single place every module gets its LLM from, so swapping providers
    (Claude -> GPT, or changing model/temperature) is a one-line change.
    """
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=temperature)
