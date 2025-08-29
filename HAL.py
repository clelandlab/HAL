import sys

def main(query):
    if "open the pod bay doors" in query.casefold():
        return "I'm sorry, Dave. I'm afraid I can't do that."
    # TODO: entry to AI is here.

sys.modules[__name__] = main
