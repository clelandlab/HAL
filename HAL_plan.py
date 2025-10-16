from google.genai import types
import json
import memory
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text

def plan(query, steps=None, silent=False):
    if not silent:
        print("[HAL] Planning...")
    docs = gather_document(f"You are giving an instruction to the next step in addressing this overall question: {query}", silent=silent)
    context = docs2text(docs)
    system_instruction = f"""You are a planner. Your task is to give the next step in answering an overall question/task.
    The following are previous completed steps:
    {steps}
    ---
    The following are relevant documents:
    {context}
    ---
    The overall question is:
    {query}
    ---
    The format of your output should be like the following:
    Step 0: a single, concise step that is independent from previous steps. Do not execute the step. The step will be given to another team member to execute. You can reference relevant documents, but do not reference any previous steps.
    Description: a brief description of the expected result.
    """
    res = memory.client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction,
        ),
        contents=query
    )
    add_generative_cost(res)
    return res.text