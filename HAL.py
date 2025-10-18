import sys
import ipywidgets as widgets
from IPython.display import display, Markdown
import memory
from HAL_sort import sort
from HAL_plan import plan
from HAL_answer import answer
from HAL_code import code, _exec, get_exec_import

_show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))

step_handlers = {}

def HAL(query=None):
    if "open the pod bay doors" in query.casefold():
        return _show("I'm sorry, Dave. I'm afraid I can't do that.")
    original_cost = memory.session.get("cost", 0)
    sequence = memory.session["sequence"]
    if query is not None:
        category = sort(query, silent=HAL.silent)
        if category == "question":
            res = answer(query, sequence, silent=HAL.silent)
            return _show(res)
        sequence.append({ "type": "user input", "input": query })
    if len(sequence) == 0:
        return _show("HAL is ready.")
    step = plan(sequence, silent=HAL.silent)
    if not HAL.silent:
        print(f"  > Step {len(sequence)}: " + step["type"])
        _show(step["prompt"])
    sequence.append(step)
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

def answer_handler(step):
    res = answer(step["prompt"], silent=HAL.silent)
    return _show(res)

step_handlers["answer"] = answer_handler

def code_handler(step):
    import_variable = { "name": HAL.name }
    c = code(step["prompt"], import_variable=import_variable, silent=HAL.silent)
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
    button = widgets.Button(description="Auto Executed" if HAL.auto_exec else "Execute", button_style='' if HAL.auto_exec else 'primary')
    button.on_click(trigger_exec)
    display(button, output)
    if HAL.auto_exec:
        trigger_exec(None)

step_handlers["code"] = code_handler

