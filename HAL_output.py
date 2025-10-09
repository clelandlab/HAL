from google import genai
from google.genai import types
import memory
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text

def output(query, silent=False):
    docs = gather_document(query, silent=silent)
    text = docs2text(docs)
    system_instruction = f"You are a researcher on experimental quantum computing. Answer the question concisely with NO comments and using ONLY the following documents:\n\n\n{text}"
    if not silent:
        print("[HAL] Thinking...")
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction
        ),
        contents=query
    )
    add_generative_cost(res)
    return res.text
