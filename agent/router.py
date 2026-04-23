"""
router_agent.py

This is the BRAIN of our agentic system.
Instead of YOUR CODE deciding which model to use (like FallbackLLM did),
NOW the AI itself reads the question and picks the best model.

How it works:
  User Question → Router AI reads it → Picks best model → That model answers
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Available models the router can choose from ───────────────────────────────
# We give the router a menu of models with descriptions
# So it knows WHAT each model is good at

AVAILABLE_MODELS = {
    "gemini": {
        "provider": "Google Gemini 🔵",
        "model":    "gemini-2.5-flash",
        "good_at":  "complex reasoning, long documents, detailed analysis",
    },
    "groq": {
        "provider": "Groq (Free ⚡)",
        "model":    "llama-3.3-70b-versatile",
        "good_at":  "fast answers, simple questions, summaries",
    },
    "openrouter": {
        "provider": "OpenRouter (Free Models 🌐)",
        "model":    "deepseek/deepseek-chat-v3-0324:free",
        "good_at":  "creative writing, general knowledge, fallback option",
    },
}


class RouterAgent:
    """
    A small AI agent that reads the user's question
    and decides which LLM model should answer it.

    It uses Groq (fast + free) to make this decision quickly.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("RouterAgent needs GROQ_API_KEY to work.")
        self.client = Groq(api_key=self.api_key)
        self.routing_log = []   # stores history of routing decisions
        print("RouterAgent initialized ✅")

    def decide_model(self, question: str) -> dict:
        """
        AI reads the question → returns which model to use.

        Returns a dict like:
        {
            "chosen_key":  "gemini",
            "provider":    "Google Gemini 🔵",
            "model":       "gemini-2.5-flash",
            "reason":      "Question needs deep analysis"
        }
        """

        # Build a prompt that explains the options to the router AI
        model_descriptions = "\n".join([
            f'- "{key}": good at {info["good_at"]}'
            for key, info in AVAILABLE_MODELS.items()
        ])

        prompt = f"""You are a routing agent. Your ONLY job is to pick the best AI model for a given question.

Available models:
{model_descriptions}

User question: "{question}"

Rules:
- If question is complex, needs deep reasoning, or is about long documents → pick "gemini"
- If question is simple, needs fast answer, or is a summary → pick "groq"  
- If question is creative or general knowledge → pick "openrouter"

Reply in this EXACT format (nothing else, no extra text):
CHOSEN: <model_key>
REASON: <one short sentence why>

Example:
CHOSEN: groq
REASON: Simple factual question that needs a fast answer.
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",   # use smallest/fastest model for routing
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,                 # 0 = deterministic, always consistent
                max_tokens=60,                   # we only need 2 lines of output
            )

            raw = response.choices[0].message.content.strip()
            print(f"[RouterAgent] Raw decision:\n{raw}")

            # Parse the response
            chosen_key, reason = self._parse_decision(raw)

            # Get full model info
            model_info = AVAILABLE_MODELS.get(chosen_key, AVAILABLE_MODELS["groq"])

            result = {
                "chosen_key": chosen_key,
                "provider":   model_info["provider"],
                "model":      model_info["model"],
                "reason":     reason,
            }

            # Log this decision
            self.routing_log.append({
                "question": question[:80] + "..." if len(question) > 80 else question,
                "decision": result,
            })

            print(f"[RouterAgent] Chose: {chosen_key} → {model_info['model']}")
            print(f"[RouterAgent] Reason: {reason}")
            return result

        except Exception as e:
            # If router itself fails, default to groq (free + fast)
            print(f"[RouterAgent] Failed to decide: {e} → defaulting to groq")
            return {
                "chosen_key": "groq",
                "provider":   AVAILABLE_MODELS["groq"]["provider"],
                "model":      AVAILABLE_MODELS["groq"]["model"],
                "reason":     "Router failed — using default (groq)",
            }

    def _parse_decision(self, raw: str) -> tuple:
        """
        Parse the router's response into (chosen_key, reason).
        Expected format:
            CHOSEN: groq
            REASON: Simple question needing fast answer.
        """
        chosen_key = "groq"   # safe default
        reason     = "Default choice"

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("CHOSEN:"):
                key = line.split(":", 1)[1].strip().lower()
                # validate it's one of our known keys
                if key in AVAILABLE_MODELS:
                    chosen_key = key
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return chosen_key, reason