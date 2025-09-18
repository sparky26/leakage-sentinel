import os
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()  # Loads .env file into environment variables

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def generate_report_streaming(analysis: dict, business_scenario: str = "") -> str:
    """
    Sends analysis + business context to Groq and returns a detailed report.
    """
    if not GROQ_AVAILABLE:
        return "Groq SDK not installed or not available."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found in environment. Please set it in .env"

    client = Groq(api_key=api_key)

    user_prompt = f"""
You are a senior data science consultant.

Dataset business context:
{business_scenario}

Dataset analysis JSON:
{json.dumps(analysis, indent=2)[:60000]}

Tasks:
1) For each feature, give a verdict: keep / review / drop, with one-line explanation.
2) For the top 5 suspicious features (by composite_score), provide:
   a) Technical justification using numeric evidence
   b) ELI5 explanation
   c) Recommended next steps (drop / transform / mask / further checks)
3) Provide an overall dataset-level summary with actionable checklist.
4) Make the report clear, structured, and human-readable.
"""
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.9,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )
        report_text = ""
        for chunk in completion:
            try:
                report_text += chunk.choices[0].delta.content or ""
            except Exception:
                pass
        return report_text
    except Exception as e:
        logger.exception("Groq API failed: %s", e)
        return f"Groq API failed: {e}"