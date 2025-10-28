from google.genai import types
import json
import memory
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, sequence2text

system_instruction = lambda docs: f"""You are a research manager leading a team. Given the step history, make a concise plan for the next step.

Your team members can access all the documents, but NOT the step history. Make sure to provide sufficient details in the prompt to make your team members work without the step history, like the detailed information from user input. Do NOT repeat document content in the prompt. Specify relevant documents so that your team members can search for them.

You may literally use an existing plan, with modification or added information. Refer to the following documents to make the plan:\n\n{docs2text(docs)}"""

def plan(sequence, silent=False):
    docs = gather_document(f"Search for documents related to high-level plans to help make plans for the next step action. Do NOT attempt to implement anything or solve the problem. Focus on high-level plans and do NOT include documents that are too detailed. If you cannot find high-level plans, search some related documents.\n\nStep history:\n\n{sequence2text(sequence)}", silent=silent)
    if not silent:
        print("[HAL] Planning...")
    config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_instruction(docs),
        response_mime_type="application/json",
        response_schema=types.Schema(type=types.Type.OBJECT, required=["prompt"], properties={
            "prompt": types.Schema(type=types.Type.STRING, description="Prompt for your team to complete the step, as a prompt for a large language model.")
        })
    )
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=config,
        contents=f"Step history:\n\n{sequence2text(sequence)}"
    )
    add_generative_cost(res)
    return json.loads(res.text)
