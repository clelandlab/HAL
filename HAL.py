import sys
import ipywidgets as widgets
from IPython.display import display, Markdown
import memory
from HAL_plan import plan
from HAL_output import output
from HAL_code import code, _exec

_show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))

step_handlers = {}

def HAL(query=None):
    if "open the pod bay doors" in query.casefold():
        return _show("I'm sorry, Dave. I'm afraid I can't do that.")
    original_cost = memory.session.get("cost", 0)
    if query is not None:
        memory.session["sequence"].append({ "type": "user input", "input": query })
    if len(memory.session["sequence"]) == 0:
        return _show("HAL is ready.")
    step = plan(memory.session["sequence"], silent=HAL.silent)
    print(f"  > Step {len(memory.session['sequence'])}: " + step["type"])
    print("  > " + step["description"])
    memory.session["sequence"].append(step)
    step_handlers[step["type"]](step)
    if not HAL.silent:
        print(f"[HAL] Cost: ${memory.session.get('cost', 0)-original_cost:.5f}. (Session Total: ${memory.session.get('cost', 0):.5f})\n")

sys.modules[__name__] = HAL

def reset():
    memory.session["sequence"] = []
    memory.session["STATE"] = {}
HAL.reset = reset

HAL.memory = memory
HAL.silent = False
HAL.auto_exec = False

HAL.name = "HAL"

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

step_handlers["output"] = lambda step: output(step["description"], silent=HAL.silent)

def code_handler(step):
    c = code(step["description"], import_variable={ "name": HAL.name }, silent=HAL.silent)
    display(Markdown(f"---\n\n```python\n{c}\n```\n\n---"))
    output = widgets.Output()
    executed = False
    def trigger_exec(b):
        nonlocal executed
        if executed:
            return
        executed = True
        with output:
            err = _exec(c, memory.session["STATE"], import_variable={ "name": HAL.name })
            if err is not None:
                print("Execution Error: ", err)
                return
    button = widgets.Button(description="Auto Executed" if HAL.auto_exec else "Execute", button_style='' if HAL.auto_exec else 'primary')
    button.on_click(trigger_exec)
    display(button, output)
    if HAL.auto_exec:
        trigger_exec(None)

step_handlers["code"] = code_handler

