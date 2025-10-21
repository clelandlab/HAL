import sys
import json
import ipywidgets as widgets
from IPython.display import display, Markdown
import memory
from HAL_sort import sort
from HAL_plan import plan
from HAL_answer import answer
from HAL_code import code, _exec, get_exec_import

_show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))

handlers = {}

def HAL(query=None):
    if query is not None and "open the pod bay doors" in query.casefold():
        return _show("I'm sorry, Dave. I'm afraid I can't do that.")
    original_cost = memory.session.get("cost", 0)
    sequence = memory.session["sequence"]
    if query is not None:
        category = sort(query, silent=HAL.silent)
        if category == "question":
            res = answer(query, sequence, silent=HAL.silent)
            return _show(res)
        sequence.append({ "user input": query })
    if len(sequence) == 0:
        return _show("HAL is ready.")
    step = plan(sequence, silent=HAL.silent)
    if not HAL.silent:
        _show(step["prompt"])
    sequence.append(step)
    handlers["code"](step)
    if not HAL.silent:
        print(f"[HAL] Cost: ${memory.session.get('cost', 0)-original_cost:.5f}. (Session Total: ${memory.session.get('cost', 0):.5f})\n")

sys.modules[__name__] = HAL

HAL.reset = lambda: memory.session.update({ "sequence": [], "STATE": {} })
HAL.save = lambda path="session.json": json.dump(memory.session, open(path, "w"), indent=2)
HAL.load = lambda path="session.json": memory.session.update(json.load(open(path, "r")))

HAL.name = "HAL"
HAL.auto = False
HAL.silent = False

HAL.memory = memory
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
    return _show(r)
HAL.search = _search

def code_handler(step):
    import_variable = { "name": HAL.name }
    c, require_input = code(step["prompt"], import_variable=import_variable, silent=HAL.silent)
    print(require_input)
    step["_code"] = c
    display(Markdown(f"---\n\n```python\n{get_exec_import(import_variable)}\n```\n\n```python\n{c}\n```\n\n---"))
    output = widgets.Output()
    executed = False
    def trigger_exec(b):
        nonlocal executed
        if executed:
            return
        executed = True
        with output:
            err = _exec(c, memory.session["STATE"], import_variable=import_variable)
            if err is not None:
                print("Execution Error: ", err)
                return
    button = widgets.Button(description="Auto Executed" if HAL.auto else "Execute", button_style='' if HAL.auto else 'primary')
    button.on_click(trigger_exec)
    display(button, output)
    if HAL.auto:
        trigger_exec(None)
handlers["code"] = code_handler

