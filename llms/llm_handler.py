"""
llm_handler.py

Contains all LLM provider classes (Gemini, Groq, OpenRouter).
The FallbackLLM has been REMOVED.
Model selection is now handled by RouterAgent in router_agent.py
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()


# ── Gemini ────────────────────────────────────────────────────────────────────
class GeminiLLM:
    MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.model_name = model_name
        self.api_key    = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Set GOOGLE_API_KEY environment variable.")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1,
        )
        print(f"Initialized Gemini LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        from langchain_core.messages import HumanMessage
        prompt = (
            f"You are a helpful AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


# ── Groq ──────────────────────────────────────────────────────────────────────
class GroqLLM:
    MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        from groq import Groq
        self.model_name = model_name
        self.api_key    = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY environment variable.")
        self.client = Groq(api_key=self.api_key)
        print(f"Initialized Groq LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        prompt = (
            f"You are a helpful AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ── OpenRouter ────────────────────────────────────────────────────────────────
class OpenRouterLLM:
    MODELS = [
        "deepseek/deepseek-chat-v3-0324:free",
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-3-27b-it:free",
    ]

    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3-0324:free", api_key: str = None):
        import requests
        self.model_name  = model_name
        self.api_key     = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable.")
        self._requests   = requests
        print(f"Initialized OpenRouter LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        prompt = (
            f"You are a helpful AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        response = self._requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       self.model_name,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=30,
        )
        data = response.json()

        if "error" in data:
            raise Exception(f"OpenRouter error: {data['error'].get('message', data['error'])}")
        if not data.get("choices"):
            raise Exception(f"OpenRouter returned no choices. Full response: {data}")

        return data["choices"][0]["message"]["content"]


# ── Provider info (used by UI / Streamlit) ────────────────────────────────────
PROVIDERS = {
    "Google Gemini 🔵":          GeminiLLM,
    "Groq (Free ⚡)":            GroqLLM,
    "OpenRouter (Free Models 🌐)": OpenRouterLLM,
}

PROVIDER_MODELS = {
    "Google Gemini 🔵":          GeminiLLM.MODELS,
    "Groq (Free ⚡)":            GroqLLM.MODELS,
    "OpenRouter (Free Models 🌐)": OpenRouterLLM.MODELS,
}

PROVIDER_ENV_KEYS = {
    "Google Gemini 🔵":          "GOOGLE_API_KEY",
    "Groq (Free ⚡)":            "GROQ_API_KEY",
    "OpenRouter (Free Models 🌐)": "OPENROUTER_API_KEY",
}

PROVIDER_LINKS = {
    "Google Gemini 🔵":          "https://aistudio.google.com/apikey",
    "Groq (Free ⚡)":            "https://console.groq.com/keys",
    "OpenRouter (Free Models 🌐)": "https://openrouter.ai/keys",
}


def create_llm(provider: str, model: str, api_key: str = None):
    """Instantiate the correct LLM class by provider name."""
    cls = PROVIDERS[provider]
    return cls(model_name=model, api_key=api_key or None)


def build_available_llms(
    groq_key:        str = None,
    gemini_key:      str = None,
    openrouter_key:  str = None,
) -> dict:
    """
    Try to initialize all available LLMs.
    Returns only the ones that succeed (key is present).

    Used by AgenticRAGPipeline:
        available_llms = build_available_llms()
        pipeline = AgenticRAGPipeline(retriever, available_llms)
    """
    available = {}

    # Groq
    try:
        available["groq"] = GroqLLM(api_key=groq_key)
    except Exception as e:
        print(f"[LLM] Groq not available: {e}")

    # Gemini
    try:
        available["gemini"] = GeminiLLM(api_key=gemini_key)
    except Exception as e:
        print(f"[LLM] Gemini not available: {e}")

    # OpenRouter
    try:
        available["openrouter"] = OpenRouterLLM(api_key=openrouter_key)
    except Exception as e:
        print(f"[LLM] OpenRouter not available: {e}")

    if not available:
        raise ValueError("No LLMs could be initialized. Check your API keys in .env")

    print(f"[LLM] Available models: {list(available.keys())}")
    return available