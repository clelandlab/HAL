from google.genai import types
import json
import memory
import config
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, evalStr

get_exec_import = lambda var: evalStr(config.EXEC_IMPORT, var)

system_instruction = lambda docs, import_variable: f"""You are a world class programming AI that generates Python code based on requirements. Write the code using ONLY the following documents:

{docs2text(docs)}

## Coding Guidelines

The code should be self-contained and runnable. Absolutely NO comments, NO explanations, NO side behaviors like printing messages.

You have an immutable global dictionary `STATE` that persists across multiple code executions. Use it to store any variables or data that need to be retained or exported. Note that you cannot assign to `STATE`, you can only modify its contents.

If any user input is necessary (e.g. missing directory path), specify them in `request_input` list. Contexts including user inputs will be passed in `STATE`. Note that all user inputs are strings.

`STATE["SIGNAL"]` is a special variable for signal. SIGNAL should be a short string in natural language, describing the key outcome of the code execution. If there is no signal description in prompt, set it to "SUCCESS" or a descriptive error message.

The following packages are already imported and ready to use. Do NOT import these packages again!

```python
{get_exec_import(import_variable)}
```"""

def code(prompt, import_variable={ "name": "HAL" }, silent=False):
    docs = gather_document(prompt, recursive=True, silent=silent)
    if not silent:
        print("[HAL] Coding...")
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["code"], properties={
                "code": types.Schema(type=types.Type.STRING),
                "request_input": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT, required=["key", "default", "description"], properties={
                    "key": types.Schema(type=types.Type.STRING),
                    "default": types.Schema(type=types.Type.STRING),
                    "description": types.Schema(type=types.Type.STRING, description="a short phrase describing the input")
                })) }),
            system_instruction=system_instruction(docs, import_variable)
        ),
        contents=prompt
    )
    add_generative_cost(res)
    r = json.loads(res.text)
    return r["code"], r.get("request_input", [])

def _exec(code, STATE={}, import_variable={ "name": "HAL" }):
    STATE["SIGNAL"] = ""
    _code = get_exec_import(import_variable) + "\n\n" + code
    try:
        exec(_code, { "STATE": STATE }, { "STATE": STATE })
        return None
    except Exception as e:
        return str(e)
