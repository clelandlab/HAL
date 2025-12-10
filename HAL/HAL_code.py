from google.genai import types
import json
from . import memory
from .HAL_gather_document import gather_document
from .utils import add_generative_cost, docs2text, get_exec_import
from .display import log

system_instruction = lambda docs, import_variable: f"""You are a world class programming AI that generates Python code based on requirements. Write the code using the given documents.

# Coding Guidelines

The code should be runnable. Absolutely NO comments, NO explanations, NO side behaviors like printing messages. Do NOT use try-except to wrap all the code, it is taken care of by the caller.

In addition to all the imported packages below, you have two global variables: `STATE` and `INVOKE`:
1. `STATE` is a dictionary that persists across multiple code executions. Use it to store any variables or data that need to be retained or exported. Note that you cannot assign to `STATE`, you can only modify its contents.
  - `STATE["SIGNAL"]` is a special variable for signal. SIGNAL should be a short string in natural language, describing the key outcome of the code execution. If there is no signal description in prompt, set it to "SUCCESS" or a descriptive error message.
2. `INVOKE` is a function that can be used to directly run other code segments or steps. `INVOKE("Code Segment [ID]")` can invoke a code segment in documents. When possible, you should use `INVOKE` instead of repeating code segments in documents.
  - Sometimes you may be instructed to invoke a number, e.g., `INVOKE(3)`, when the manager decides to run a previous step. Faithfully follow the instruction to invoke the specified step.

If any user input is necessary (e.g. missing directory path), specify them in `request_input` list. `request_input` should be a code snippet that assigns values to variables in `STATE`. It will be modified by the user to input the necessary values.

# Documents

{docs2text(docs)}

# Imports

The following packages are already imported and ready to use. Do NOT import these packages again!

```python
{get_exec_import(import_variable)}
```"""

def code(prompt, import_variable={ "name": "HAL" }, _doc={}):
    docs = gather_document(prompt)
    _doc["code"] = list(map(lambda d: d["id"], docs))
    log("[HAL] Coding...", "Coding")
    res = memory.client.models.generate_content(
        model="gemini-3-pro-preview",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["code"], properties={
                "code": types.Schema(type=types.Type.STRING),
                "request_input": types.Schema(type=types.Type.STRING, description="some lines of code assigning values to variables in STATE. This will be modified by the user.")
            }),
            system_instruction=system_instruction(docs, import_variable)
        ),
        contents=prompt
    )
    add_generative_cost(res)
    r = json.loads(res.text)
    return r["code"], r.get("request_input")
