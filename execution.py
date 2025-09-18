def _exec(code):
    loc = {}
    exec(code, globals(), loc)
    return loc
