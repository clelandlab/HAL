from google.genai import types
import json
from . import memory
from .utils import add_generative_cost
from .display import log

system_instruction = '''You are an expert at classifying user prompts into categories. Given a user prompt, classify it into one of the following categories:

- "query": if the prompt is asking for an answer, explanation, or natural language response, like **how to** write a function, why something is wrong, or to write a document.
- "action": if the prompt is requesting to perform a task or action, like take a measurement or run a data analysis or fix something.
'''

def sort(prompt):
    log("[HAL] Sorting...", "Sorting")
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=types.Schema(type=types.Type.OBJECT, required=["category"], properties={ "category": types.Schema(type=types.Type.STRING) })
    )
    res = memory.client.models.generate_content(
        model="gemini-flash-latest",
        config=config,
        contents=prompt
    )
    category = json.loads(res.text)["category"]
    log(f"  > {category}")
    return category
