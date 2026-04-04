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
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Set GOOGLE_API_KEY environment variable.")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1
        )
        print(f"Initialized Gemini LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        from langchain_core.messages import HumanMessage
        prompt = (
            f"You are a helpful AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"


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
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY environment variable.")
        self.client = Groq(api_key=self.api_key)
        print(f"Initialized Groq LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        prompt = (
            f"You are a helpful AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"


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
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable.")
        self._requests = requests
        print(f"Initialized OpenRouter LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        prompt = (
            f"You are a helpful AI assistant. Use the context below to answer accurately.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        try:
            response = self._requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=60,
            )
            data = response.json()

            if "error" in data:
                return f"OpenRouter error: {data['error'].get('message', data['error'])}"

            if not data.get("choices"):
                return f"OpenRouter returned no choices. Full response: {data}"

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error generating response: {e}"


# ── Factory ───────────────────────────────────────────────────────────────────
PROVIDERS = {
    "Google Gemini 🔵": GeminiLLM,
    "Groq (Free ⚡)": GroqLLM,
    "OpenRouter (Free Models 🌐)": OpenRouterLLM,
}

PROVIDER_MODELS = {
    "Google Gemini 🔵": GeminiLLM.MODELS,
    "Groq (Free ⚡)": GroqLLM.MODELS,
    "OpenRouter (Free Models 🌐)": OpenRouterLLM.MODELS,
}

PROVIDER_ENV_KEYS = {
    "Google Gemini 🔵": "GOOGLE_API_KEY",
    "Groq (Free ⚡)": "GROQ_API_KEY",
    "OpenRouter (Free Models 🌐)": "OPENROUTER_API_KEY",
}

PROVIDER_LINKS = {
    "Google Gemini 🔵": "https://aistudio.google.com/apikey",
    "Groq (Free ⚡)": "https://console.groq.com/keys",
    "OpenRouter (Free Models 🌐)": "https://openrouter.ai/keys",
}


def create_llm(provider: str, model: str, api_key: str = None):
    """Instantiate the correct LLM class."""
    cls = PROVIDERS[provider]
    return cls(model_name=model, api_key=api_key or None)


# ── Advanced RAG Pipeline ─────────────────────────────────────────────────────
class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history: List[Dict[str, Any]] = []

    def swap_llm(self, new_llm):
        """Hot-swap the LLM without rebuilding the whole pipeline."""
        self.llm = new_llm

    def query(self, question: str, top_k: int = 5, min_score: float = 0.0) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)

        if not results:
            answer = "No relevant context found in your documents."
            sources = []
        else:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [
                {
                    "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
                    "page": doc["metadata"].get("page", "unknown"),
                    "score": doc["similarity_score"],
                    "preview": doc["content"][:120] + "...",
                }
                for doc in results
            ]
            answer = self.llm.generate_response(question, context)

        citations = [f"[{i+1}] {s['source']} (page {s['page']})" for i, s in enumerate(sources)]
        answer_with_citations = answer + "\n\n**Sources:**\n" + "\n".join(citations) if citations else answer

        self.history.append({"question": question, "answer": answer, "sources": sources})

        return {"question": question, "answer": answer_with_citations, "sources": sources, "history": self.history}