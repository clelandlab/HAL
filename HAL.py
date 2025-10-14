import sys
import ipywidgets as widgets
from IPython.display import display, Markdown
import memory
from HAL_output import output
from HAL_code import code, _exec

_show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))

def HAL(query, t=""):
    if "open the pod bay doors" in query.casefold():
        return _show("I'm sorry, Dave. I'm afraid I can't do that.")
    memory.session["step"] = [] # reset steps
    original_cost = memory.session.get("cost", 0)
    res = output(query, silent=HAL.silent)
    if not HAL.silent:
        print(f"[HAL] Cost: ${memory.session.get('cost', 0)-original_cost:.5f}. (Session Total: ${memory.session.get('cost', 0):.5f})\n")
    return _show(res)

HAL.memory = memory
HAL.silent = False
HAL.auto_exec = False

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

sys.modules[__name__] = HAL

# Temporary functions:
def _code(query, STATE={}):
    c = code(query)
    display(Markdown(f"---\n\n```python\n{c}\n```\n\n---"))
    output = widgets.Output()
    executed = False
    def trigger_exec(b):
        nonlocal executed
        if executed:
            return
        executed = True
        with output:
            err = _exec(c, STATE)
            if err is not None:
                print("Execution Error: ", err)
                return
    button = widgets.Button(description="Auto Executed" if HAL.auto_exec else "Execute", button_style='' if HAL.auto_exec else 'primary')
    button.on_click(trigger_exec)
    display(button, output)
    if HAL.auto_exec:
        trigger_exec(None)

HAL.code = _code

