from google import genai
from google.genai import types
import json
import memory
import config
from HAL_gather_document import gather_document
from utils import client, add_generative_cost, docs2text, evalStr

get_exec_import = lambda var: evalStr(config.EXEC_IMPORT, var)

def code(query, import_variable={ "name": "HAL" }, silent=False):
    docs = gather_document(query, silent=silent)
    text = docs2text(docs)
    system_instruction = f"You are a world class programming AI that generates Python code based on user requirements. Write the code using ONLY the following documents:\n\n\n{text}\n\n\nThe code should be self-contained and runnable. Do NOT include any side behaviors like printing messages. Absolutely do NOT include any comments or explanations.\nTo export data or variables for later use, save them to the global dictionary `STATE`. The following packages are already imported (do NOT import them again!):\n{get_exec_import(import_variable)}"
    if not silent:
        print("[HAL] Coding...")
    res = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=types.Schema(type=types.Type.OBJECT, required=["code"], properties={ "code": types.Schema(type=genai.types.Type.STRING) }),
            system_instruction=system_instruction
        ),
        contents=query
    )
    add_generative_cost(res)
    return json.loads(res.text)["code"]

def _exec(code, STATE={}, import_variable={ "name": "HAL" }):
    _code = get_exec_import(import_variable) + "\n\n" + code
    try:
        exec(_code, { "STATE": STATE }, { "STATE": STATE })
        return None
    except Exception as e:
        return str(e)
