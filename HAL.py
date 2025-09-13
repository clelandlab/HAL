import sys, ai, memory
from IPython.display import display, Markdown

show = lambda x: display(Markdown(x))

def HAL(query):
    if "open the pod bay doors" in query.casefold():
        return show("I'm sorry, Dave. I'm afraid I can't do that.")
    res = ai.gen(query, silent=HAL.silent)
    return show(res)

HAL.memory = memory
HAL.ai = ai
HAL.silent = False

sys.modules[__name__] = HAL
