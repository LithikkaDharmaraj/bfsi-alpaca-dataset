import ollama

MODEL_NAME = "phi3:mini"
# MODEL_NAME = "Qwen2.5-1.5B"


def generate_response(prompt: str) -> str:
    """
    Sends structured prompt to local SLM via Ollama
    and returns generated response.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0.3
        }
    )

    return response["message"]["content"]