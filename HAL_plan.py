from google.genai import types
import json
import memory
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, sequence2text

def plan(sequence, steps=None, silent=False):
    if not silent:
        print("[HAL] Planning...")
    docs = gather_document(f"Search for documents related to high-level plans to help make plans for the next step. Do NOT attempt to implement anything or solve the problem. Do NOT include documents that are too detailed.\n\nStep history:\n\n{sequence2text(sequence)}", silent=silent)
    docs_text = docs2text(docs)
    return
    # TODO: finalize the prompt
    system_instruction = f"""You are a planner. Your task is to give the next step in answering an overall question/task.
    The following are previous completed steps. Using ONLY the following documents:\n\n{docs_text}"""
    config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_instruction
    )
    # TODO: structured output
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=config,
        contents=f"Step history:\n\n{sequence2text(sequence)}"
    )
    add_generative_cost(res)
    return res.text
