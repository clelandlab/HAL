import sys, ai, memory, execution
from IPython.display import display, Markdown

_show = lambda x: display(Markdown(x))

def HAL(query, t=""):
    if "open the pod bay doors" in query.casefold():
        return _show("I'm sorry, Dave. I'm afraid I can't do that.")
    original_cost = memory.session.get("cost", 0)
    docs = ai.gather_document(query, silent=HAL.silent)
    res = ai.question(query, docs, silent=HAL.silent)
    if not HAL.silent:
        print(f"[HAL] Cost: ${memory.session.get('cost', 0)-original_cost:.5f}. (Total: {memory.session.get('cost', 0):.5f})\n")
    return _show(res)

HAL.memory = memory
HAL.ai = ai
HAL.silent = False

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
        r += f"\n\n---\n\n{doc['content']}\n\n\n"
    return _show(r)
HAL.search = _search

HAL.exec_import = '''import quick, skynet, time, os, sys, json, yaml
import numpy as np
import matplotlib.pyplot as plt
skynet.label = ""
'''
HAL.exec_code = ""
HAL.exec = lambda: execution._exec(HAL.exec_import + "\n" + HAL.exec_code)

sys.modules[__name__] = HAL
