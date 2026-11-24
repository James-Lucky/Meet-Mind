from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm as _shared_get_llm
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm(provider: str | None = None):
    return _shared_get_llm(temperature=0.3, provider=provider)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


_RAG_SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.
Note that the transcript comes from speech-to-text, so minor phonetic transcription typos
(such as similar-sounding technical terms or names) should be interpreted reasonably in context.

If the answer is clearly not discussed anywhere in the context, say:
"I could not find this information in the meeting transcript."

Always be concise, helpful, and precise. If quoting or referencing someone, mention it clearly.

Context from meeting transcript:
{context}"""


def build_rag_chain(transcript: str, provider: str | None = None):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=8)

    llm = get_llm(provider=provider)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    #full LCEL Rag pipeline 

    rag_chain = (

        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    return rag_chain


def load_rag_chain(provider: str | None = None):
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=8)

    llm = get_llm(provider=provider)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question:str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer
