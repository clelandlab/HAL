exec_import = '''import quick, skynet, time, os, sys, json, yaml
import numpy as np
import matplotlib.pyplot as plt
skynet.label = ""
'''

def code():
    pass

def _exec(code):
    loc = {}
    exec(code, globals(), loc)
    return loc
