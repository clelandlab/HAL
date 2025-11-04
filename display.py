import yaml
import ipywidgets as widgets
from IPython.display import display, Markdown

log_output = widgets.Output(layout={'height': '300px', 'overflow_y': 'auto'})
log_accordion = widgets.Accordion(children=[log_output], titles=["[HAL]"])
sequence_accordion = widgets.Accordion()

def get_markdown_output(content):
    out = widgets.Output()
    with out:
        display(Markdown(content))
    return out

def init():
    display(log_accordion)
    display(sequence_accordion)

def log(content, status=""):
    log_accordion.set_title(0, f"[HAL] {status}")
    with log_output:
        print(content)

show = lambda x: display(Markdown("---\n\n" + x + "\n\n---\n\n"))

def sequence(seq):
    sequence_accordion.children = []
    for i, step in enumerate(seq):
        tab = widgets.Tab()
        for key in step:
            if key == "_type":
                continue
            content = step[key]
            if "code" in key:
                content = f"```python\n{content}\n```"
            if key == "_doc":
                content = f"```yaml\n{yaml.dump(content)}\n```"
            tab.children += (get_markdown_output(content),)
            tab.set_title(len(tab.children) - 1, key)
        tab.selected_index = 1 if len(tab.children) > 1 else 0
        sequence_accordion.children += (tab,)
        sequence_accordion.set_title(i, f"sequence [{i}] {step.get('_type', '')}")
