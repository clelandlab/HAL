from google.genai import types
import json
import memory
import config
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, evalStr

get_exec_import = lambda var: evalStr(config.EXEC_IMPORT, var)

def code(prompt, import_variable={ "name": "HAL" }, silent=False):
    docs = gather_document(prompt, silent=silent)
    system_instruction = f"You are a world class programming AI that generates Python code based on user requirements. Write the code using ONLY the following documents:\n\n{docs2text(docs)}\n\n\nThe code should be self-contained and runnable. Do NOT include any side behaviors like printing messages. Absolutely do NOT include any comments or explanations.\nIf any user input is necessary, specify them in `require_input` list. User inputs will be passed as a global dictionary `STATE`.\nTo export data or variables for later use, save them to the global dictionary `STATE` (you cannot assign to `STATE`, you can only update it). The following packages are already imported (do NOT import them again!):\n{get_exec_import(import_variable)}"
    if not silent:
        print("[HAL] Coding...")
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["code"], properties={
                "code": types.Schema(type=types.Type.STRING),
                "require_input": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT, required=["key", "description"], properties={
                    "key": types.Schema(type = types.Type.STRING),
                    "default": types.Schema(type = types.Type.STRING),
                    "description": types.Schema(type = types.Type.STRING)
                })) }),
            system_instruction=system_instruction
        ),
        contents=prompt
    )
    add_generative_cost(res)
    r = json.loads(res.text)
    return r["code"], r.get("require_input", [])

def _exec(code, STATE={}, import_variable={ "name": "HAL" }):
    _code = get_exec_import(import_variable) + "\n\n" + code
    try:
        exec(_code, { "STATE": STATE }, { "STATE": STATE })
        return None
    except Exception as e:
        return str(e)
