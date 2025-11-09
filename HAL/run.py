import re
from . import memory
from .utils import get_exec_import

def execute(code, import_variable={ "name": "HAL" }):
    STATE = memory.session["STATE"]
    STATE["SIGNAL"] = ""
    _code = get_exec_import(import_variable) + "\n\n" + code
    INVOKE = lambda name: invoke(name, import_variable=import_variable)
    exec(_code, { "STATE": STATE, "INVOKE": INVOKE }, { "STATE": STATE, "INVOKE": INVOKE })

def invoke(name, import_variable={ "name": "HAL" }):
    code = ""
    if isinstance(name, int):
        step = memory.session["sequence"][name]
        code = step.get("_code", "")
    if isinstance(name, str):
        docs = memory.search(name, filter=lambda d: "invoke" in d and d["invoke"])
        if len(docs) == 0:
            return
        doc = memory.get(docs[0][0])
        code_match = re.search(r"```python\n(.*?)\n```", doc["content"], re.DOTALL)
        code = code_match.group(1) if code_match else ""
    if code == "":
        return
    execute(code, import_variable=import_variable)
