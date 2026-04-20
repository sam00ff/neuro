def echo(text: str):
    return {
        "ok": True,
        "result": f"You said: {text}"
    }


def get_tools():
    return {
        "echo": echo
    }