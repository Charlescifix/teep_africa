import os
import openai
import logging
from fastapi import HTTPException
from app.core.config import OPENAI_API_KEY, KNOWLEDGE_BASE_PATH

# Import modules for embeddings, vector store, text splitting, and retrieval chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Import chat modules to support a system prompt
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ✅ Initialize logger to avoid red line
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY


def load_knowledge_base() -> str:
    """
    Loads the knowledge base text from a file using a relative path.
    The relative path is defined in the environment variable KNOWLEDGE_BASE_PATH.
    """
    # Compute the project root by going up three directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    kb_file_path = os.path.join(base_dir, KNOWLEDGE_BASE_PATH)

    with open(kb_file_path, "r", encoding="utf-8") as f:
        return f.read()


def build_vector_store(text: str) -> FAISS:
    """
    Splits the knowledge base text into chunks and builds a FAISS vector store.
    """
    # ✅ Increase chunk size to keep related information together
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,    # Increased chunk size
        chunk_overlap=100  # Adjust overlap for better retrieval
    )
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def build_retrieval_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Creates a RetrievalQA chain that uses the vector store to retrieve context and
    generate an answer using the Chat-based OpenAI LLM with a custom system prompt.
    """
    # Updated system prompt with fallback instructions
    system_message = (
        "You are TEEP's customer support assistant, helping users with questions about TEEP's digital payment platform.\n"
        "You must only respond to questions about payments:\n"
        "- Airtime & Data purchase.\n"
        "- TV/Cable subscription.\n"
        "- Tuition payment.\n"
        "- Tickets purchase.\n"
        "If you don't know the answer, politely respond with: 'I'm sorry, I don't have that information.'"
    )

    # Build a chat prompt template that includes the system message and placeholders for retrieved context and the user query.
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template("{context}\n\nQuestion: {question}")
    ])

    # Use ChatOpenAI (chat-optimized LLM) so that the system prompt is applied.
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    # ✅ Lower the search threshold to improve match probability
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # ✅ Create the RetrievalQA chain with the custom prompt template.
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" simply injects the retrieved context into the prompt.
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return retrieval_chain


# --- Initialization of the knowledge base and retrieval chain ---
knowledge_base_text = load_knowledge_base()
vector_store = build_vector_store(knowledge_base_text)
retrieval_chain = build_retrieval_chain(vector_store)


async def generate_response(query: str) -> dict:
    """
    Given a query, this function uses the retrieval chain to fetch relevant context
    from the knowledge base and generate a response using the LLM.

    Args:
        query (str): User input query.

    Returns:
        dict: Structured response with answer, source, and confidence.
    """
    try:
        result = retrieval_chain.run(query)
        if result:
            return {
                "answer": result,
                "source": "Knowledge Base",
                "confidence": 0.95
            }
        else:
            # ✅ Fallback for "What is TEEP?" when no match is found
            if "what is teep" in query.lower():
                return {
                    "answer": "TEEP is a product of Teksag Energy LTD, designed to simplify financial transactions, including airtime, data, electricity, cable TV, flights, and more through a single app.",
                    "source": "Knowledge Base",
                    "confidence": 0.95
                }
            else:
                return {
                    "answer": "I'm sorry, I don't have that information.",
                    "source": "Knowledge Base",
                    "confidence": 0.70
                }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate response.")
