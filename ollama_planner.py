import requests
import json
import re

def ollama_planner(goal, tools):
    prompt = f"""
You are an AI planner.

Goal: {goal}

Available tools:
{list(tools.keys())}

Return a JSON plan like:
[
  {{
    "tool": "tool_name",
    "args": {{}}
  }}
]

ONLY return JSON. No explanation.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    output = response.json().get("response", "")

    # 🔥 Try direct JSON parse
    try:
        plan = json.loads(output)
        return plan
    except:
        print("⚠️ Raw Ollama output:")
        print(output)

    # 🔥 Fallback: extract JSON from text
    match = re.search(r'\[.*\]', output, re.DOTALL)

    if match:
        try:
            plan = json.loads(match.group())
            return plan
        except:
            return []

    return []