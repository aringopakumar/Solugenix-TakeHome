"""
Question-Answering Chain Module
================================
Combines the retriever (vector store) with an LLM to answer user
questions grounded in company documents.

Architecture:
  User Question
       │
       ▼
  Retriever  ──► top-k most relevant chunks
       │
       ▼
  Prompt Template  ──► combines question + retrieved context
       │
       ▼
  LLM (Llama via Ollama)  ──► generates a grounded answer

The chain also returns source documents so the UI can display
which parts of the knowledge base were used.
"""

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


# ------------------------------------------------------------------
# Prompt template – instructs the LLM to answer ONLY from context
# ------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a helpful FAQ assistant for a company.
Use ONLY the following context to answer the user's question.
If the answer is not contained in the context, say:
"I'm sorry, I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer (be concise and helpful):"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_qa_chain(
    vectorstore: FAISS,
    model_name: str = "llama3.2",
    temperature: float = 0.0,
    k: int = 6,
) -> RetrievalQA:
    """
    Build a RetrievalQA chain that:
      1. Retrieves the top-k most relevant chunks for a question.
      2. Passes them as context to the LLM.
      3. Returns the answer AND the source documents.

    Parameters
    ----------
    vectorstore : FAISS
        The vector store to retrieve from.
    model_name : str
        Ollama model to use (e.g. "llama3.2", "mistral").
    temperature : float
        LLM temperature. 0.0 = deterministic, factual answers.
    k : int
        Number of chunks to retrieve per question.

    Returns
    -------
    RetrievalQA
        A callable chain. Usage: chain.invoke({"query": "..."})
    """
    # Initialize the LLM
    llm = ChatOllama(model=model_name, temperature=temperature)

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 4},
    )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # "stuff" = concatenate all chunks into one prompt
        retriever=retriever,
        return_source_documents=True, # Include source docs in the response
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain
