import os
from typing import Any, List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI


class MistralChatModel(BaseChatModel):
    """Lightweight LangChain chat wrapper for the Mistral Chat Completions API."""

    model: str = "mistral-small-latest"
    api_key: str
    temperature: float = 0.3
    timeout: int = 120
    api_url: str = "https://api.mistral.ai/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    @staticmethod
    def _convert_message(msg: BaseMessage) -> dict:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "user"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {"role": role, "content": content}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(m) for m in messages],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"Mistral API error ({response.status_code}) for model '{self.model}': {response.text}"
            )

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _build_gemini_llms(temperature: float) -> list:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return []

    primary_model = (os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest").strip()
    models = [primary_model]
    for fallback_model in ("gemini-2.5-flash-lite", "gemini-2.5-flash"):
        if fallback_model not in models:
            models.append(fallback_model)

    return [
        ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_retries=1,
        )
        for model_name in models
    ]


def _build_mistral_llms(temperature: float) -> list:
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    if not api_key:
        return []

    primary_model = (os.getenv("MISTRAL_MODEL") or "mistral-small-latest").strip()
    models = [primary_model]
    for fallback_model in ("open-mistral-nemo", "mistral-large-latest"):
        if fallback_model not in models:
            models.append(fallback_model)

    return [
        MistralChatModel(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )
        for model_name in models
    ]


def get_llm(temperature: float = 0.3, provider: Optional[str] = None):
    """
    Build a LangChain chat runnable with automatic cross-provider fallback:
    - 'auto' or 'gemini' (default): tries Gemini first, then falls back to Mistral
    - 'mistral': tries Mistral first, then falls back to Gemini
    """
    load_dotenv(override=True)
    selected = (provider or os.getenv("LLM_PROVIDER") or "auto").strip().lower()

    gemini_llms = _build_gemini_llms(temperature)
    mistral_llms = _build_mistral_llms(temperature)

    if selected in ("auto", "gemini"):
        llms = gemini_llms + mistral_llms
    elif selected == "mistral":
        llms = mistral_llms + gemini_llms
    else:
        raise ValueError(f"Unsupported LLM provider '{selected}'. Expected 'auto', 'gemini', or 'mistral'.")

    if not llms:
        raise ValueError("Neither GEMINI_API_KEY nor MISTRAL_API_KEY is set in .env")

    if len(llms) == 1:
        return llms[0]
    return llms[0].with_fallbacks(llms[1:])

