from google.genai import types
import json
from . import memory
from .HAL_gather_document import gather_document
from .utils import add_generative_cost, docs2text, sequence2text
from .display import log

system_instruction = lambda docs: f"""You are a research manager leading a team. Given the step history, make a concise plan for the next step.

Your team members can access all the documents, but NOT the step history. Make sure to provide sufficient details in the prompt to make your team members work without the step history, like the detailed information from user input.

Do NOT repeat document content in the prompt. Specify relevant documents title or keyword so that your team members can search for them. It's not recommended to refer to plan documents unless necessary. Do NOT use document indices, they are not accessible to your team members.

SIGNAL is a special string variable that describes the key outcome of the step. It can include some critical numbers like goodness of fitting, or short messages like "SUCCESS" or error. Provide a SIGNAL description in your prompt, so that your team can present the result to you. 

If you want to repeat a step, execute a step for multiple times, or execute a code segment, ask your team to INVOKE it by its step index or the code segment name in the prompt.

If the task requested by the user is completed, set the step type to "end" and output empty prompt.

You may literally use an existing plan, with modification or added information. Refer to the following documents to make the plan:\n\n{docs2text(docs)}"""

def plan(sequence, _doc={}):
    docs = gather_document(f'Make a plan for the next step. Do NOT attempt to implement anything or solve the problem. I only need documents with "Plan" explicitly in the title.\n\nStep history:\n\n{sequence2text(sequence)}')
    _doc["plan"] = list(map(lambda d: d["id"], docs))
    log("[HAL] Planning...", "Planning")
    config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_instruction(docs),
        response_mime_type="application/json",
        response_schema=types.Schema(type=types.Type.OBJECT, required=["type", "prompt"], properties={
            "type": types.Schema(type=types.Type.STRING, description="Type of the next step, one of: 'code', 'end'."),
            "prompt": types.Schema(type=types.Type.STRING, description="Prompt for your team to complete the step, as a prompt for a large language model.")
        })
    )
    res = memory.client.models.generate_content(
        model="gemini-2.5-pro",
        config=config,
        contents=f"Step history:\n\n{sequence2text(sequence)}"
    )
    add_generative_cost(res)
    return json.loads(res.text)
