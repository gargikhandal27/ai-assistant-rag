"""
tools.py

These are the TOOLS our agent can use.
Think of tools like superpowers the AI can choose to use.

Tools we have:
  1. search_web   → searches the internet for latest information
  2. send_email   → sends an email with the answer

The agent DECIDES on its own:
  - "Do I need to search the web for this?"
  - "Should I send this answer by email?"
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Web Search
# ══════════════════════════════════════════════════════════════════════════════

def search_web(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Search the internet using Tavily (free, much better than DuckDuckGo).
    Works for weather, news, prices, current events — everything.
    """
    print(f"[WebSearch] Searching for: '{query}'")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "success": False,
            "results": [],
            "query":   query,
            "message": "TAVILY_API_KEY not set in .env file.",
        }

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":        api_key,
                "query":          query,
                "max_results":    num_results,
                "search_depth":   "basic",   # "basic" is free, "advanced" is paid
            },
            timeout=10,
        )
        data = response.json()

        results = []
        for r in data.get("results", []):
            results.append({
                "title":   r.get("title", "Result"),
                "snippet": r.get("content", ""),
                "url":     r.get("url", ""),
            })

        if results:
            print(f"[WebSearch] Found {len(results)} result(s)")
            return {"success": True, "results": results, "query": query}
        else:
            return {"success": False, "results": [], "query": query,
                    "message": "No results found."}

    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return {"success": False, "results": [], "query": query,
                "message": f"Search failed: {str(e)}"}


def format_web_results(search_output: Dict[str, Any]) -> str:
    """
    Convert raw search results into clean text
    so the LLM can use it as context.
    """
    if not search_output["success"] or not search_output["results"]:
        return f"Web search for '{search_output['query']}' returned no results."

    lines = [f"Web search results for: '{search_output['query']}'\n"]
    for i, r in enumerate(search_output["results"], 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['snippet']}")
        if r.get("url"):
            lines.append(f"   Source: {r['url']}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Email Sender
# ══════════════════════════════════════════════════════════════════════════════

def send_email(
    to_email:    str,
    subject:     str,
    body:        str,
    from_email:  str = None,
    app_password: str = None,
) -> Dict[str, Any]:
    """
    Send an email with the AI's answer using Gmail SMTP.

    Setup needed (one time only):
      1. Go to myaccount.google.com → Security → App Passwords
      2. Create an app password for "Mail"
      3. Add to your .env file:
            EMAIL_ADDRESS=your@gmail.com
            EMAIL_APP_PASSWORD=your_app_password

    Returns:
    {
        "success": True/False,
        "message": "Email sent!" or error message
    }
    """

    # Read from .env if not passed directly
    from_email   = from_email   or os.environ.get("EMAIL_ADDRESS")
    app_password = app_password or os.environ.get("EMAIL_APP_PASSWORD")

    # Validate we have what we need
    if not from_email:
        return {
            "success": False,
            "message": "EMAIL_ADDRESS not set in .env file.",
        }
    if not app_password:
        return {
            "success": False,
            "message": "EMAIL_APP_PASSWORD not set in .env file.",
        }

    print(f"[EmailTool] Sending email to: {to_email}")

    try:
        # Build the email
        msg = MIMEMultipart()
        msg["From"]    = from_email
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Connect to Gmail and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())

        print(f"[EmailTool] ✅ Email sent successfully to {to_email}")
        return {
            "success": True,
            "message": f"Email sent successfully to {to_email}",
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": (
                "Gmail authentication failed. "
                "Make sure EMAIL_APP_PASSWORD is correct. "
                "Go to myaccount.google.com → Security → App Passwords"
            ),
        }
    except Exception as e:
        print(f"[EmailTool] Error: {e}")
        return {
            "success": False,
            "message": f"Failed to send email: {str(e)}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY — list of all tools the agent can see and use
# ══════════════════════════════════════════════════════════════════════════════

# This is like a menu card for the agent
# Agent reads this and decides which tool to call

TOOL_REGISTRY = {
    "search_web": {
        "function":    search_web,
        "description": (
            "Search the internet for current information. "
            "Use when question is about recent events, news, or info not in PDFs."
        ),
        "when_to_use": [
            "current events",
            "recent news",
            "today",
            "latest",
            "price",
            "score",
            "not in documents",
        ],
    },
    "send_email": {
        "function":    send_email,
        "description": (
            "Send the answer by email. "
            "Use when user asks to email the answer or share results."
        ),
        "when_to_use": [
            "send email",
            "email me",
            "share by email",
            "mail this",
            "send to",
        ],
    },
}