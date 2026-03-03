from google.genai import types
import json
from . import memory
from .HAL_gather_document import gather_document
from .utils import add_generative_cost, docs2text, state_type2text, get_exec_import
from .display import log

system_instruction = lambda docs, import_variable, STATE: f"""You are a world class programming AI that generates Python code based on requirements. Write clear and concise code using the given documents.

# Coding Guidelines

The code should be runnable. Absolutely NO comments, NO explanations, NO side behaviors like printing messages. Do NOT use try-except to wrap all the code, it is taken care of by the caller. If any user input is really necessary (e.g. missing data directory), specify them in `request_input`, which should be a code snippet that assigns values to variables in `STATE`. It will be modified by the user to input the necessary values.


In addition to all the imported packages below, you have two global variables: `STATE` and `INVOKE`:
1. `STATE` is a dictionary that persists across steps. Use it to store any variables or data that need to be retained or exported. Note that you cannot assign to `STATE`, you can only modify its contents.
  - `STATE["SIGNAL"]` is a special variable for signal. SIGNAL should be a short string in natural language, describing the key outcome of the code execution. If there is no signal description in prompt, set it to "SUCCESS" or a descriptive error message.
2. `INVOKE` is a function that can be used to directly run other code segments or steps. `INVOKE("Code Segment [ID]")` can invoke a code segment in documents. When possible, you should use `INVOKE` instead of repeating code segments in documents.
  - Sometimes you may be instructed to invoke a number, e.g., `INVOKE(3)`, when the manager decides to run a previous step. Faithfully follow the instruction to invoke the specified step.

## Existing Variables in STATE

Take the following variables as given. Do NOT check or request user input! Not every variable is relevant to your task. Only use the specified or relevant variables.

{state_type2text(STATE)}

## Documents

{docs2text(docs)}

## Imports

The following packages are already imported and ready to use. Do NOT import these packages again!

```python
{get_exec_import(import_variable)}
```"""

def code(prompt, import_variable={ "name": "HAL" }, _doc={}):
    docs = gather_document(prompt)
    _doc["code"] = list(map(lambda d: d["id"], docs))
    model = memory.session.get("model", "gemini-flash-latest")
    log(f"[HAL] Coding ({model})...", "Coding")
    res = memory.client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["code"], properties={
                "code": types.Schema(type=types.Type.STRING),
                "request_input": types.Schema(type=types.Type.STRING, description="some lines of code assigning values to variables in STATE. ONLY assignment statements are allowed. This will be modified by the user.")
            }),
            system_instruction=system_instruction(docs, import_variable, memory.session["STATE"])
        ),
        contents=prompt
    )
    add_generative_cost(res)
    r = json.loads(res.text)
    request_input = r.get("request_input")
    if request_input == "":
        request_input = None
    return r["code"], request_input
