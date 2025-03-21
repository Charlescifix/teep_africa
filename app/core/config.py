import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.txt")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL")


print(OPENAI_API_KEY)
print(DATABASE_URL)
print(KNOWLEDGE_BASE_PATH)