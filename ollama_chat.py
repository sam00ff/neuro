import requests

def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        data = response.json()

        # ✅ Handle multiple formats safely
        if "response" in data:
            return data["response"].replace("\\n", "\n")

        elif "message" in data and "content" in data["message"]:
            return data["message"]["content"].replace("\\n", "\n")

        else:
            return str(data)

    except Exception as e:
        return f"Ollama error: {str(e)}"