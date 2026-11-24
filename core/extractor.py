#Actionableitems , decision , questions 

from langchain_core.prompts import ChatPromptTemplate   
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm as _shared_get_llm


def get_llm(provider: str | None = None):
    return _shared_get_llm(temperature=0.2, provider=provider)


def build_chain(system_prompt: str, provider: str | None = None):
    llm = get_llm(provider=provider)
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript: str, provider: str | None = None) -> str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        provider=provider,
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str, provider: str | None = None) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        provider=provider,
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str, provider: str | None = None) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        provider=provider,
    )
    return chain.invoke(transcript)