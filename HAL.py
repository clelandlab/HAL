import sys, ai, memory, execution
from IPython.display import display, Markdown

_show = lambda x: display(Markdown(x))

def HAL(query, t=""):
    if "open the pod bay doors" in query.casefold():
        return show("I'm sorry, Dave. I'm afraid I can't do that.")
    docs = ai.gather_document(query, silent=HAL.silent)
    if t == "code":
        HAL.exec_code = ai.code(query, docs, exec_import=HAL.exec_import, silent=HAL.silent)
        return _show(f"```python\n{HAL.exec_import}\n```\n\n```python\n{HAL.exec_code}\n```")
    res = ai.question(query, docs, silent=HAL.silent)
    return _show(res)

HAL.memory = memory
HAL.ai = ai
HAL.silent = False

def _search(*args, **kwargs):
    res, scores = memory.search(*args, **kwargs)
    r = ""
    for doc, score in zip(res, scores):
        r += f"### `{doc["id"]}`\n- **score**: {score}\n"
        for k, v in doc.items():
            if k in ["id", "content", "embedding"]:
                continue
            r += f"- **{k}**: {v}\n"
        r += f"\n\n---\n\n{doc['content']}\n\n\n"
    return _show(r)
HAL.search = _search

HAL.exec_import = """import quick, skynet, time, os, sys, json
import numpy as np
import matplotlib.pyplot as plt"""
HAL.exec_code = ""
HAL.exec = lambda: execution._exec(HAL.exec_import + "\n\n" + HAL.exec_code)

sys.modules[__name__] = HAL
