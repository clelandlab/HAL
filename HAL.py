import sys, os, json, random, string
from google import genai
import ipywidgets as widgets
from IPython.display import display as _display
import memory, utils, display
from HAL_sort import sort
from HAL_plan import plan
from HAL_answer import answer
from HAL_code import code, _exec, get_exec_import

handlers = {}

def HAL(query=None):
    if query is not None and "open the pod bay doors" in query.casefold():
        return display.show("I'm sorry, Dave. I'm afraid I can't do that.")
    original_cost = memory.session.get("cost", 0)
    sequence = memory.session["sequence"]
    if query is not None:
        category = sort(query)
        if category == "question":
            res = answer(query, sequence)
            display.log("")
            return display.show(res)
        sequence.append({ "user input": query, "_type": "user input" })
    if len(sequence) == 0:
        return display.show("HAL is ready.")
    display.sequence(sequence)
    res = plan(sequence)
    step = { "prompt": res["prompt"] }
    step["_type"] = res["type"]
    display.log(f"  > {step['_type']}")
    sequence.append(step)
    display.sequence(sequence)
    handlers[step["_type"]](step)
    display.log(f"[HAL] Cost: ${memory.session.get('cost', 0)-original_cost:.5f}. (Session Total: ${memory.session.get('cost', 0):.5f})\n")
    display.sequence(sequence)

sys.modules[__name__] = HAL

HAL.name = "HAL"
HAL.auto = False
HAL.memory = memory

# Following are HAL methods

def _init(name, _config=None):
    HAL.name = name
    if _config is None:
        _config = os.path.dirname(os.path.abspath(__file__)) + '/config.json'
    if isinstance(_config, dict):
        utils.config.update(_config)
    if isinstance(_config, str):
        utils.config.update(json.load(open(_config, "r")))
    memory.client = genai.Client(api_key=utils.config["GEMINI_API_KEY"])
    display.init()
    memory.load()
    display.log("[HAL] Ready.")
HAL.init = _init

def _reset():
    memory.session.update({ "cost": 0.0, "sequence": [], "STATE": {} })
    display.log("[HAL] Session reset.")
    display.sequence(memory.session.get("sequence", []))
HAL.reset = _reset

def _save(path="session.json"):
    display.log(f"[HAL] Session saved to {path}")
    return json.dump(memory.session, open(path, "w"), indent=2)
HAL.save = _save

def _load(path="session.json"):
    display.log(f"[HAL] Session loaded from {path}")
    memory.session.update(json.load(open(path, "r")))
    display.sequence(memory.session.get("sequence", []))
HAL.load = _load

def _search(*args, **kwargs):
    res = memory.search(*args, **kwargs)
    r = ""
    for id, score in res:
        doc = memory.get(id)
        r += f"### `{doc["id"]}`\n- **score**: {score}\n"
        for k, v in doc.items():
            if k in ["id", "content", "embedding"]:
                continue
            r += f"- **{k}**: {v}\n"
        r += f"\n\n{doc['content']}\n\n---\n\n"
    return display.show(r)
HAL.search = _search

def _memorize(content=None, meta={}):
    if HAL.name == "HAL":
        return print("[HAL] Error: Please set HAL.name to memorize")
    if "source" not in meta:
        meta["source"] = HAL.name
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
handlers["end"] = end_handler

def code_handler(step):
    import_variable = { "name": HAL.name }
    c, request_input = code(step["prompt"], import_variable=import_variable)
    input_widgets = {}
    step["_code"] = c
    display.show(f"```python\n{get_exec_import(import_variable)}\n```\n\n```python\n{c}\n```")
    for v in request_input:
        print(f'- input: {v["description"]}')
        w = widgets.Text(value=v.get("default", ""), description=v["key"])
        input_widgets[v["key"]] = w
        _display(w)
    output = widgets.Output()
    def trigger_exec(b):
        with output:
            for k in input_widgets:
                memory.session["STATE"][k] = input_widgets[k].value
            err = _exec(c, memory.session["STATE"], import_variable=import_variable)
            step["SIGNAL"] = memory.session["STATE"].get("SIGNAL", "")
            if err is not None:
                print("Execution Error: ", err)
                return
    auto_exec = HAL.auto and len(request_input) == 0
    button = widgets.Button(description="Auto Executed" if auto_exec else "Execute", button_style='' if auto_exec else 'primary')
    button.on_click(trigger_exec)
    _display(button, output)
    if auto_exec:
        trigger_exec(None)
handlers["code"] = code_handler

