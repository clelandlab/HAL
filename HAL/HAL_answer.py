from google.genai import types
from . import memory
from .HAL_gather_document import gather_document
from .utils import add_generative_cost, docs2text, sequence2text
from .display import log

def answer(prompt, sequence):
    docs = gather_document(prompt)
    system_instruction = f"You are a researcher. Answer the question concisely with NO comments. Use the provided context and the following documents (you might refer to document title, but NOT document number):\n\n{docs2text(docs)}"
    model = memory.session.get("model", "gemini-flash-latest")
    log(f"[HAL] Answering ({model})...", "Answering")
    res = memory.client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=f"Context:\n\n{sequence2text(sequence)}\n\nQuestion:\n\n{prompt}"
    )
    add_generative_cost(res)
    return res.text
