from google.genai import types
import json
from . import memory
from .utils import add_generative_cost
from .display import log

system_instruction = '''You are an expert at classifying user prompts into categories. Given a user prompt, classify it into one of the following categories:

- "question": if the prompt is asking for an answer or explanation, like how to write a code or why something is wrong.
- "action": if the prompt is requesting to perform a task or action, like take a measurement or run a data analysis or fix something.
'''

def sort(prompt):
    log("[HAL] Sorting...", "Sorting")
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=types.Schema(type=types.Type.OBJECT, required=["category"], properties={ "category": types.Schema(type=types.Type.STRING) })
    )
    res = memory.client.models.generate_content(
        model="gemini-2.5-flash",
        config=config,
        contents=prompt
    )
    category = json.loads(res.text)["category"]
    log(f"  > {category}")
    return category
