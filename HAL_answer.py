from google.genai import types
import memory
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, sequence2text

def answer(prompt, sequence, silent=False):
    docs = gather_document(prompt, silent=silent)
    system_instruction = f"You are a researcher on experimental quantum computing. Answer the question concisely with NO comments and using ONLY the provided context and the following documents:\n\n{docs2text(docs)}"
    if not silent:
        print("[HAL] Answering...")
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction
        ),
        contents=f"Context:\n\n{sequence2text(sequence)}\n\nQuestion:\n\n{prompt}"
    )
    add_generative_cost(res)
    return res.text
