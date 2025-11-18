from google.genai import types
from . import memory
from .HAL_gather_document import gather_document
from .utils import add_generative_cost, docs2text, sequence2text
from .display import log

def answer(prompt, sequence):
    docs = gather_document(prompt)
    system_instruction = f"You are a researcher on experimental quantum computing. Answer the question concisely with NO comments and using ONLY the provided context and the following documents:\n\n{docs2text(docs)}"
    log("[HAL] Answering...", "Answering")
    res = memory.client.models.generate_content(
        model="gemini-3-pro-preview",
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=f"Context:\n\n{sequence2text(sequence)}\n\nQuestion:\n\n{prompt}"
    )
    add_generative_cost(res)
    return res.text
