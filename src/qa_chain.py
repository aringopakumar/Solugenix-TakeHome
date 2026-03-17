"""Builds the RetrievalQA chain that ties the vector store to the LLM."""

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


PROMPT_TEMPLATE = """You are a helpful FAQ assistant for a company.
Use ONLY the following context to answer the user's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

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
    """Create a chain that retrieves top-k chunks and sends them to the LLM."""
    llm = ChatOllama(model=model_name, temperature=temperature)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 4},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain
