import sys, ai, memory
from IPython.display import display, Markdown

_show = lambda x: display(Markdown(x))

def HAL(query):
    if "open the pod bay doors" in query.casefold():
        return show("I'm sorry, Dave. I'm afraid I can't do that.")
    res = ai.gen(query, silent=HAL.silent)
    return _show(res)

HAL.memory = memory
HAL.ai = ai
HAL.silent = False

def _search(*args, **kwargs):
    res, scores = memory.search(*args, **kwargs)
    r = ""
    for doc, score in zip(res, scores):
        r += f"### `{doc["id"]}`\n- **Score**: {score}\n"
        for k, v in doc.items():
            if k in ["id", "content", "embedding"]:
                continue
            r += f"- **{k}**: {v}\n"
        r += f"\n\n---\n\n{doc['content']}\n\n"
    return _show(r)
HAL.search = _search

sys.modules[__name__] = HAL
