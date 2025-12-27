import sys, os, json, random, string, time
from google import genai
import ipywidgets as widgets
from IPython.display import display as _display
from . import memory, utils, display, run
from .HAL_gather_document import gather_document
from .HAL_sort import sort
from .HAL_plan import plan
from .HAL_answer import answer
from .HAL_code import code

handlers = {}

def HAL(query=None):
    if query is not None and "open the pod bay doors" in query.casefold():
        return display.show("I'm sorry, Dave. I'm afraid I can't do that.")
    while True:
        original_cost = memory.session.get("cost", 0)
        log_cost = lambda: display.log(f"[HAL] Cost: ${memory.session.get('cost', 0)-original_cost:.5f}. (Session Total: ${memory.session.get('cost', 0):.5f})\n")
        start_time = time.time()
        sequence = memory.session["sequence"]
        if query is not None:
            category = sort(query)
            if category == "query":
                res = answer(query, sequence)
                log_cost()
                return display.show(res)
            sequence.append({ "user input": query, "_type": "user input" })
        if len(sequence) == 0:
            return display.show("HAL is ready.")
        if sequence[-1].get("_type", "") == "end":
            return display.show("HAL session has ended. Please reset the session using `HAL.reset()`.")
        if "SIGNAL" in memory.session["STATE"]:
            sequence[-1]["SIGNAL"] = memory.session["STATE"]["SIGNAL"]
            del memory.session["STATE"]["SIGNAL"]
        display.sequence(sequence)
        step = { "_doc": {} }
        res = plan(sequence, _doc=step["_doc"])
        step["_type"], step["prompt"] = res["type"], res["prompt"]
        display.log(f"  > {step['_type']}")
        sequence.append(step)
        display.sequence(sequence)
        pause = handlers[step["_type"]](step)
        display.log(f"[HAL] Step time: {time.time()-start_time:.2f} s")
        log_cost()
        display.sequence(sequence)
        query = None
        if pause:
            return display.show("HAL sequence is paused.")
        if HAL.auto <= 0 or step["_type"] == "end":
            HAL.auto = 0
            return display.show("HAL auto is stopped.")
        HAL.auto -= 1

sys.modules[__name__] = HAL

HAL.auto = 0
HAL.session = memory.session

HAL.memory = memory
HAL.display = display

HAL.gather_document = gather_document
HAL.sort = sort
HAL.plan = plan
HAL.answer = answer
HAL.code = code

_invoke = lambda name=None, import_variable={}: run.invoke(name, import_variable={ **memory.session, **import_variable })

def _export_ctx():
    main_namespace = sys.modules.get('__main__')
    main_namespace.STATE = memory.session["STATE"]
    main_namespace.INVOKE = _invoke
    HAL.session = memory.session

def _init(name, _config=None):
    memory.session["name"] = name
    if _config is None:
        _config = os.path.join(os.path.dirname(__file__), "../config.json")
    if isinstance(_config, dict):
        utils.config.update(_config)
    if isinstance(_config, str):
        utils.config.update(json.load(open(_config, "r")))
    memory.client = genai.Client(api_key=utils.config["GEMINI_API_KEY"])
    display.init()
    memory.load()
    display.log("[HAL] Initialized.")
    HAL.reset()
HAL.init = _init

def _reset():
    memory.session.update({ "cost": 0.0, "sequence": [], "STATE": {} })
    display.log("[HAL] Session reset.")
    display.sequence(memory.session["sequence"])
    _export_ctx()
HAL.reset = _reset

def _save(path="session.json"):
    display.log(f"[HAL] Session saved to {path}")
    return json.dump(memory.session, open(path, "w"), indent=2)
HAL.save = _save

def _load(path="session.json"):
    display.log(f"[HAL] Session loaded from {path}")
    memory.session.update(json.load(open(path, "r")))
    display.sequence(memory.session.get("sequence", []))
    _export_ctx()
HAL.load = _load

def _search(*args, **kwargs):
    res = memory.search(*args, **kwargs)
    texts = []
    for id, score in res:
        doc = memory.get(id)
        r = f"### `{doc["id"]}`\n- **score**: {score}\n"
        for k, v in doc.items():
            if k in ["id", "content", "embedding"]:
                continue
            r += f"- **{k}**: {v}\n"
        r += f"\n\n{doc['content']}\n\n"
        texts.append(r)
    return display.docs(texts)
HAL.search = _search

def _memorize(content=None, meta={}):
    if memory.session["name"] == "HAL":
        return print("[HAL] Error: Please set name to memorize")
    if "source" not in meta:
        meta["source"] = memory.session["name"]
    if isinstance(content, str):
        return memory.add(content, meta)
    if content is None:
        content = len(memory.session.get("sequence", [])) - 1
    if isinstance(content, int):
        seq = memory.session.get("sequence", [])
        if content < 0 or content >= len(seq):
            return print(f"[HAL] Error: Invalid sequence [{content}]")
        step = seq[content]
        if "prompt" not in step or "_code" not in step:
            return print(f"[HAL] Error: Sequence [{content}] does not contain a valid step to memorize")
        n = "".join(random.choices(string.ascii_uppercase, k=2)) + str(len(memory.data.keys()))
        c = f"# Code Segment {n}:\n\n## Prompt:\n\n{step['prompt']}\n\n## Code:\n\nYou can directly run the following code by calling `INVOKE('Code Segment {n}')`\n\n```python\n{step['_code']}\n```"
        meta["invoke"] = 1
        return memory.add(c, meta)
    return print("[HAL] Error: Unsupported content type for memorize")
HAL.memorize = _memorize

def end_handler(step):
    step["prompt"] = "Session ended."
    display.show("[HAL] session ended.")
    return False
handlers["end"] = end_handler

def code_handler(step):
    STATE = memory.session["STATE"]
    c, request_input = code(step["prompt"], import_variable=memory.session, _doc=step["_doc"])
    step["_code"] = c
    display.sequence(memory.session["sequence"])
    if HAL.auto and (request_input is None):
        display.log(f"[HAL] Executing...", "Executing")
        try:
            run.execute(c, import_variable=memory.session)
            print("Execution Completed with SIGNAL: ", STATE.get("SIGNAL", ""))
        except Exception as err:
            STATE["SIGNAL"] = f"Runtime Error: {type(e).__name__}: {str(err)}"
            print("Execution Error: ", str(err))
        display.log(f"[HAL] Execution completed with SIGNAL: {STATE['SIGNAL']}")
        display.sequence(memory.session["sequence"])
        return False
    display.new_cell((f"# [HAL] Requesting user input:\n{request_input}\n\n" if request_input is not None else "") + f"# [HAL] Imports:\n{utils.get_exec_import(memory.session)}\n\n# [HAL] Code:\n{c}")
    return True
handlers["code"] = code_handler

