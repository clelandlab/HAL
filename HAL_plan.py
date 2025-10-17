from google.genai import types
import json
import memory
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, sequence2text

system_instruction = lambda docs_text: f"""You are a research manager leading a team. Given the step history, make a concise plan for the next step.

If you want to provide an answer without executing any code, choose the 'output' type.
If you need to implement and execute Python code to proceed, choose the 'code' type.

Your team member will NOT access the step history. Make sure to provide sufficient details in the description to make your team members work without the step history. Your team members have access to all the documents. Do NOT repeat document content in the plan, you may specify the keyword of the document so that your team members can search for it. Using ONLY the following documents:\n\n{docs_text}"""

def plan(sequence, steps=None, silent=False):
    docs = gather_document(f"Search for documents related to high-level plans to help make plans for the next step. Do NOT attempt to implement anything or solve the problem. Do NOT include documents that are too detailed.\n\nStep history:\n\n{sequence2text(sequence)}", silent=silent)
    docs_text = docs2text(docs)
    if not silent:
        print("[HAL] Planning...")
    config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_instruction(docs_text),
        response_mime_type="application/json",
        response_schema=types.Schema(type=types.Type.OBJECT, required=["type", "description"], properties={
            "type": types.Schema(type=types.Type.STRING, description="The type of the next step, one of: 'code'(implement and execute Python code), 'output'(answer questions or output results to the user)"),
            "description": types.Schema(type=types.Type.STRING, description="A detailed description of the next step to take. Write the description as a prompt for large language model.")
        })
    )
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=config,
        contents=f"Step history:\n\n{sequence2text(sequence)}"
    )
    add_generative_cost(res)
    return json.loads(res.text)
